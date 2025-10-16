# app.R — Adaptive Decision Analysis Shiny App (Replay animations)
# ---------------------------------------------------------------
# New:
#  • Replay tab: build animations from saved history using plotly
#  • Animations: Regret over time, Pulls-by-Arm over time, Success vs Failure over time

if (!requireNamespace("shiny", quietly = TRUE)) install.packages("shiny")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("bslib", quietly = TRUE)) install.packages("bslib")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")
if (!requireNamespace("purrr", quietly = TRUE)) install.packages("purrr")
if (!requireNamespace("DT", quietly = TRUE)) install.packages("DT")
if (!requireNamespace("scales", quietly = TRUE)) install.packages("scales")
if (!requireNamespace("plotly", quietly = TRUE)) install.packages("plotly")

library(shiny)
library(ggplot2)
library(bslib)
library(dplyr)
library(tidyr)
library(purrr)
library(DT)
library(scales)
library(plotly)

# ---------- Helpers ----------
rbern <- function(n, p) rbinom(n, 1, p)

ucb1_index <- function(mean_reward, pulls, t) {
  ifelse(pulls == 0, Inf, mean_reward + sqrt(2 * log(pmax(1, t)) / pulls))
}
thompson_index <- function(alpha, beta) rbeta(length(alpha), alpha, beta)
choose_epsilon_greedy <- function(Q, eps) if (runif(1) < eps) sample(seq_along(Q), 1) else which.max(Q)

step_bandit <- function(state, policy, params) {
  t <- state$t + 1
  arm <- switch(policy,
                "Thompson"       = { idx <- thompson_index(state$alpha, state$beta); which.max(idx) },
                "UCB1"           = { Q <- state$rewards_sum / pmax(1, state$pulls); idx <- ucb1_index(Q, state$pulls, t); which.max(idx) },
                "Epsilon-Greedy" = { Q <- state$rewards_sum / pmax(1, state$pulls); choose_epsilon_greedy(Q, params$epsilon) }
  )
  r <- rbern(1, state$true_p[arm])
  
  state$pulls[arm]       <- state$pulls[arm] + 1
  state$rewards_sum[arm] <- state$rewards_sum[arm] + r
  state$alpha[arm]       <- state$alpha[arm] + r
  state$beta[arm]        <- state$beta[arm]  + (1 - r)
  state$t <- t
  
  state$history <- bind_rows(
    state$history,
    tibble(
      step = t, arm = arm, reward = r,
      cum_reward    = sum(state$rewards_sum),
      best_possible = t * max(state$true_p),
      regret        = (t * max(state$true_p)) - sum(state$rewards_sum)
    )
  )
  state
}

init_state <- function(true_p, prior_alpha = 1, prior_beta = 1) {
  list(
    true_p      = true_p,
    pulls       = rep(0, length(true_p)),
    rewards_sum = rep(0, length(true_p)),
    alpha       = rep(prior_alpha, length(true_p)),
    beta        = rep(prior_beta,  length(true_p)),
    t           = 0,
    history     = tibble(step = integer(), arm = integer(), reward = integer(),
                         cum_reward = integer(), best_possible = double(), regret = double())
  )
}

# Build cumulative "pulls by arm" over time from history
build_pulls_over_time <- function(h, K) {
  if (nrow(h) == 0) return(tibble())
  # cumulative counts per arm at each step
  # For each step s, count pulls of each arm up to s
  base <- expand_grid(step = seq_len(max(h$step)), arm = 1:K)
  counts <- h %>%
    count(step, arm) %>%
    complete(step = seq_len(max(h$step)), arm = 1:K, fill = list(n = 0)) %>%
    arrange(arm, step) %>%
    group_by(arm) %>%
    mutate(pulls = cumsum(n)) %>%
    ungroup() %>%
    select(step, arm, pulls)
  base %>% left_join(counts, by = c("step","arm")) %>% mutate(pulls = replace_na(pulls, 0))
}

# Build cumulative hits/misses per arm over time
build_hits_misses_over_time <- function(h, K) {
  if (nrow(h) == 0) return(tibble())
  # For each arm, cumulative hits and misses up to each step
  hits <- h %>%
    group_by(step, arm) %>%
    summarize(hits_step = sum(reward), .groups = "drop") %>%
    complete(step = seq_len(max(h$step)), arm = 1:K, fill = list(hits_step = 0)) %>%
    arrange(arm, step) %>%
    group_by(arm) %>%
    mutate(Hit = cumsum(hits_step)) %>%
    ungroup()
  
  pulls <- h %>%
    group_by(step, arm) %>%
    summarize(pulls_step = n(), .groups = "drop") %>%
    complete(step = seq_len(max(h$step)), arm = 1:K, fill = list(pulls_step = 0)) %>%
    arrange(arm, step) %>%
    group_by(arm) %>%
    mutate(Pulls = cumsum(pulls_step)) %>%
    ungroup()
  
  hm <- hits %>% left_join(pulls, by = c("step","arm")) %>%
    mutate(Miss = pmax(0, Pulls - Hit)) %>%
    select(step, arm, Hit, Miss)
  
  hm %>%
    pivot_longer(cols = c(Hit, Miss), names_to = "Outcome", values_to = "Count") %>%
    mutate(arm = factor(arm))
}

# ---------- UI ----------
ui <- page_fluid(
  theme = bs_theme(version = 5, bootswatch = "flatly"),
  title = "Adaptive Decision Analysis — Bandit Playground",
  navset_tab(
    nav_panel("Learn (1-Arm)",
              layout_sidebar(
                sidebar = sidebar(
                  h4("Single-Arm Bernoulli"),
                  checkboxInput("reveal_p", "Reveal true p", FALSE),
                  conditionalPanel(
                    condition = "input.reveal_p == true",
                    sliderInput("true_p", "True success probability:",
                                min = 0.05, max = 0.95, value = 0.65, step = 0.01)
                  ),
                  hr(),
                  sliderInput("a0", "Prior alpha (Beta):", min = 0.1, max = 10, value = 1, step = 0.1),
                  sliderInput("b0", "Prior beta (Beta):",  min = 0.1, max = 10, value = 1, step = 0.1),
                  numericInput("n_pulls", "Pulls per click:", value = 1, min = 1, step = 1),
                  actionButton("pull1", "Pull the Arm"),
                  actionButton("reset1", "Reset")
                ),
                card(
                  card_header("Posterior & Data"),
                  fluidRow(
                    column(6, plotOutput("post_plot", height = 300)),
                    column(6, tableOutput("sum_tbl"), verbatimTextOutput("ci_text"))
                  )
                )
              )
    ),
    
    nav_panel("Play (Multi-Arm)",
              layout_sidebar(
                sidebar = sidebar(
                  h4("Configure"),
                  numericInput("K", "Number of arms:", value = 4, min = 2, max = 10, step = 1),
                  uiOutput("arm_prob_ui"),
                  checkboxInput("randomize", "Randomize arm probabilities", TRUE),
                  checkboxInput("reveal_true_multi", "Reveal true arm probabilities", FALSE),
                  hr(),
                  radioButtons("policy", "Policy:", c("Thompson", "UCB1", "Epsilon-Greedy"), inline = TRUE),
                  sliderInput("epsilon", "ε for ε-Greedy:", min = 0, max = 0.5, value = 0.1, step = 0.01),
                  numericInput("horizon", "Horizon (rounds):", value = 100, min = 10, max = 1000, step = 10),
                  helpText("Note: Larger horizons take longer to simulate."),
                  actionButton("step_once", "Step"),
                  actionButton("auto_run", "Auto Run"),
                  actionButton("reset_play", "Reset")
                ),
                card(
                  card_header("Simulation"),
                  fluidRow(
                    column(6, plotOutput("regret_plot", height = 300)),
                    column(6, plotOutput("pulls_plot", height = 300))
                  ),
                  fluidRow(
                    column(12, plotOutput("sf_plot", height = 320))
                  ),
                  fluidRow(
                    column(12, DTOutput("arm_summary_tbl"))
                  )
                )
              )
    ),
    
    # NEW: Replay tab
    nav_panel("Replay",
              layout_sidebar(
                sidebar = sidebar(
                  h4("Build Animations from Last Run"),
                  helpText("After running on the Play tab, click below to generate animated replays."),
                  actionButton("build_replay", "Build Replay"),
                  helpText("Tip: increase Horizon on Play for longer animations.")
                ),
                card(
                  card_header("Regret Replay"),
                  plotlyOutput("regret_anim", height = 360)
                ),
                card(
                  card_header("Pulls-by-Arm Replay"),
                  plotlyOutput("pulls_anim", height = 360)
                ),
                card(
                  card_header("Success vs Failure Replay"),
                  plotlyOutput("sf_anim", height = 420)
                )
              )
    ),
    
    nav_panel("Compare",
              layout_sidebar(
                sidebar = sidebar(
                  h4("Replicate & Compare"),
                  numericInput("rep_K", "Number of arms:", value = 4, min = 2, max = 10),
                  checkboxInput("rep_randomize", "Randomize arm probabilities", TRUE),
                  uiOutput("rep_arm_prob_ui"),
                  numericInput("rep_horizon", "Horizon:", value = 300, min = 50, step = 50),
                  numericInput("rep_n", "Replicates per policy:", value = 100, min = 10, step = 10),
                  sliderInput("rep_eps", "ε for ε-Greedy:", min = 0, max = 0.5, value = 0.1, step = 0.01),
                  actionButton("run_compare", "Run Comparison")
                ),
                card(
                  card_header("Cumulative Regret (mean ± 95% CI)"),
                  plotOutput("compare_plot", height = 360),
                  DTOutput("compare_tbl"),
                  uiOutput("compare_hint")
                )
              )
    )
  )
)

# ---------- Server ----------
server <- function(input, output, session) {
  # ===== Tab 1: Single-Arm =====
  rv1 <- reactiveValues(a = 1, b = 1, s = 0, f = 0, hidden_p = runif(1, 0.05, 0.95))
  observeEvent(input$reset1, {
    rv1$a <- input$a0; rv1$b <- input$b0; rv1$s <- 0; rv1$f <- 0
    rv1$hidden_p <- runif(1, 0.05, 0.95)
  })
  true_p_single <- reactive({
    if (isTRUE(input$reveal_p) && !is.null(input$true_p)) input$true_p else rv1$hidden_p
  })
  observeEvent(input$pull1, {
    isolate({
      s <- sum(rbern(input$n_pulls, true_p_single()))
      f <- input$n_pulls - s
      rv1$s <- rv1$s + s
      rv1$f <- rv1$f + f
      rv1$a <- input$a0 + rv1$s
      rv1$b <- input$b0 + rv1$f
    })
  })
  output$post_plot <- renderPlot({
    a <- rv1$a; b <- rv1$b
    x <- seq(0.001, 0.999, length.out = 500)
    df <- data.frame(x = x, dens = dbeta(x, a, b))
    ggplot(df, aes(x, dens)) +
      geom_line(linewidth = 1) +
      labs(x = "p", y = "Beta density", title = "Posterior Beta(a, b)") +
      theme_minimal(base_size = 12)
  })
  output$sum_tbl <- renderTable({
    tibble(
      prior_alpha     = input$a0,
      prior_beta      = input$b0,
      successes       = rv1$s,
      failures        = rv1$f,
      posterior_alpha = rv1$a,
      posterior_beta  = rv1$b,
      posterior_mean  = rv1$a / (rv1$a + rv1$b),
      true_p_revealed = ifelse(isTRUE(input$reveal_p), round(true_p_single(), 3), NA)
    )
  })
  output$ci_text <- renderText({
    a <- rv1$a; b <- rv1$b
    ci <- qbeta(c(0.025, 0.975), a, b)
    paste0("95% credible interval for p: [", round(ci[1], 3), ", ", round(ci[2], 3), "]")
  })
  
  # ===== Tab 2: Play (fast sim; plots update per click/auto batches) =====
  output$arm_prob_ui <- renderUI({
    if (isTRUE(input$randomize)) return(NULL)
    lapply(seq_len(input$K), function(k) {
      sliderInput(paste0("armp_", k), paste0("Arm ", k, " true p:"),
                  min = 0.05, max = 0.95, value = round(runif(1, 0.1, 0.9), 2), step = 0.01)
    })
  })
  true_p_vec <- reactive({
    K <- input$K
    if (isTRUE(input$randomize)) sort(runif(K, 0.05, 0.95)) else {
      v <- sapply(seq_len(K), function(k) input[[paste0("armp_", k)]])
      if (any(is.null(v))) sort(runif(K, 0.05, 0.95)) else v
    }
  })
  rv2 <- reactiveValues(state = NULL, auto = FALSE)
  observeEvent(input$reset_play, {
    rv2$state <- init_state(true_p_vec(), prior_alpha = 1, prior_beta = 1)
    rv2$auto  <- FALSE
  }, priority = 10)
  observeEvent(true_p_vec(), {
    rv2$state <- init_state(true_p_vec(), prior_alpha = 1, prior_beta = 1)
    rv2$auto  <- FALSE
  })
  observeEvent(input$step_once, {
    req(rv2$state)
    rv2$state <- step_bandit(rv2$state, input$policy, list(epsilon = input$epsilon))
  })
  # Auto-run in small batches so UI remains responsive
  autoInvalidate <- reactiveTimer(75)
  observe({
    if (rv2$auto) {
      autoInvalidate()
      if (!is.null(rv2$state) && nrow(rv2$state$history) < input$horizon) {
        steps_to_do <- min(5, input$horizon - nrow(rv2$state$history))
        for (i in seq_len(steps_to_do)) {
          rv2$state <- step_bandit(rv2$state, input$policy, list(epsilon = input$epsilon))
        }
      } else {
        rv2$auto <- FALSE
      }
    }
  })
  observeEvent(input$auto_run, { rv2$auto <- TRUE })
  
  output$regret_plot <- renderPlot({
    req(rv2$state)
    h <- rv2$state$history
    if (nrow(h) == 0) return(NULL)
    p <- ggplot(h, aes(step, regret))
    if (nrow(h) > 1) p <- p + geom_line(linewidth = 1) else p <- p + geom_point(size = 2)
    p + labs(title = "Cumulative Regret", x = "Step", y = "Regret") +
      theme_minimal(base_size = 12)
  })
  output$pulls_plot <- renderPlot({
    req(rv2$state)
    h <- rv2$state$history
    if (nrow(h) == 0) return(NULL)
    K <- length(rv2$state$true_p)
    pulls_by_arm <- h %>% count(arm) %>% complete(arm = 1:K, fill = list(n = 0))
    ggplot(pulls_by_arm, aes(factor(arm), n)) +
      geom_col() +
      labs(title = "Pulls by Arm", x = "Arm", y = "Count") +
      theme_minimal(base_size = 12)
  })
  output$sf_plot <- renderPlot({
    req(rv2$state)
    st <- rv2$state
    K  <- length(st$true_p)
    hits   <- st$rewards_sum
    pulls  <- st$pulls
    misses <- pmax(0, pulls - hits)
    df <- tibble(
      arm = factor(1:K),
      Hit = hits,
      Miss = misses
    ) %>% pivot_longer(cols = c(Hit, Miss), names_to = "Outcome", values_to = "Count")
    ggplot(df, aes(arm, Count, fill = Outcome)) +
      geom_col() +
      labs(title = "Success (Hit) vs Failure (Miss) by Arm", x = "Arm", y = "Count") +
      theme_minimal(base_size = 12)
  })
  output$arm_summary_tbl <- renderDT({
    req(rv2$state)
    st <- rv2$state
    K  <- length(st$true_p)
    pulls  <- st$pulls
    hits   <- st$rewards_sum
    misses <- pmax(0, pulls - hits)
    emp_mean  <- ifelse(pulls > 0, hits / pulls, NA_real_)
    post_mean <- (st$alpha) / (st$alpha + st$beta)
    pct_pulls <- if (sum(pulls) > 0) pulls / sum(pulls) else rep(0, K)
    df <- tibble(
      Arm = 1:K,
      Pulls = pulls,
      Hits = hits,
      Misses = misses,
      `Empirical mean` = round(emp_mean, 3),
      `Posterior mean` = round(post_mean, 3),
      `% of pulls` = sprintf("%.1f%%", 100 * pct_pulls)
    )
    if (isTRUE(input$reveal_true_multi)) df <- df %>% mutate(`True p` = round(st$true_p, 3))
    datatable(df, rownames = FALSE, options = list(pageLength = 10))
  })
  
  # ===== Tab 3: Replay (build animations from saved history) =====
  replay_data <- reactiveVal(NULL)
  
  observeEvent(input$build_replay, {
    req(rv2$state)
    h <- rv2$state$history
    req(nrow(h) > 0)
    K <- length(rv2$state$true_p)
    
    pulls_time <- build_pulls_over_time(h, K)
    hm_time    <- build_hits_misses_over_time(h, K)
    
    replay_data(list(
      history = h,
      pulls_time = pulls_time,
      hm_time = hm_time
    ))
  })
  
  output$regret_anim <- renderPlotly({
    rd <- replay_data(); if (is.null(rd)) return(NULL)
    h <- rd$history
    # Use cumulative line with frame = step to show progression
    # Build a frame-wise data where each frame shows line up to that step
    frames <- lapply(seq_len(max(h$step)), function(s) {
      d <- h %>% filter(step <= s)
      d$frame <- s
      d
    }) %>% bind_rows()
    
    plot_ly(frames, x = ~step, y = ~regret, frame = ~frame, type = 'scatter', mode = 'lines') %>%
      layout(title = "Cumulative Regret (Replay)", xaxis = list(title = "Step"), yaxis = list(title = "Regret")) %>%
      animation_opts(frame = 50, easing = "linear", redraw = TRUE) %>%
      animation_slider(currentvalue = list(prefix = "Step "))
  })
  
  output$pulls_anim <- renderPlotly({
    rd <- replay_data(); if (is.null(rd)) return(NULL)
    d <- rd$pulls_time %>% mutate(arm = factor(arm))
    plot_ly(d, x = ~arm, y = ~pulls, frame = ~step, type = 'bar') %>%
      layout(barmode = "group", title = "Pulls by Arm (Replay)",
             xaxis = list(title = "Arm"), yaxis = list(title = "Cumulative pulls")) %>%
      animation_opts(frame = 50, easing = "linear", redraw = TRUE) %>%
      animation_slider(currentvalue = list(prefix = "Step "))
  })
  
  output$sf_anim <- renderPlotly({
    rd <- replay_data(); if (is.null(rd)) return(NULL)
    d <- rd$hm_time
    plot_ly(d, x = ~arm, y = ~Count, color = ~Outcome, frame = ~step, type = 'bar') %>%
      layout(barmode = "stack", title = "Success vs Failure by Arm (Replay)",
             xaxis = list(title = "Arm"), yaxis = list(title = "Cumulative count")) %>%
      animation_opts(frame = 50, easing = "linear", redraw = TRUE) %>%
      animation_slider(currentvalue = list(prefix = "Step "))
  })
  
  # ===== Tab 4: Compare (unchanged summary) =====
  output$rep_arm_prob_ui <- renderUI({
    if (isTRUE(input$rep_randomize)) return(NULL)
    lapply(seq_len(input$rep_K), function(k) {
      sliderInput(paste0("repp_", k), paste0("Arm ", k, " true p:"),
                  min = 0.05, max = 0.95, value = round(runif(1, 0.1, 0.9), 2), step = 0.01)
    })
  })
  rep_true_p <- reactive({
    K <- input$rep_K
    if (isTRUE(input$rep_randomize)) sort(runif(K, 0.05, 0.95)) else {
      v <- sapply(seq_len(K), function(k) input[[paste0("repp_", k)]])
      if (any(is.null(v))) sort(runif(K, 0.05, 0.95)) else v
    }
  })
  run_one <- function(policy, horizon, true_p, eps = 0.1) {
    st <- init_state(true_p)
    for (t in seq_len(horizon)) st <- step_bandit(st, policy, list(epsilon = eps))
    tail(st$history$regret, 1)
  }
  comp_out <- reactiveVal(NULL)
  observeEvent(input$run_compare, {
    comp_out(NULL)
    withProgress(message = "Running simulations...", value = 0, {
      H <- input$rep_horizon; N <- input$rep_n; tp <- rep_true_p()
      pols <- c("Thompson", "UCB1", "Epsilon-Greedy")
      all <- map_dfr(seq_along(pols), function(i) {
        pol <- pols[i]
        vals <- replicate(N, run_one(pol, H, tp, eps = input$rep_eps))
        incProgress(1/length(pols), detail = paste("Policy:", pol))
        tibble(policy = pol, regret = vals)
      })
      comp_out(all)
    })
  })
  output$compare_plot <- renderPlot({
    out <- comp_out(); if (is.null(out)) return(NULL)
    summ <- out %>% group_by(policy) %>% summarize(
      mean_regret = mean(regret),
      lo = quantile(regret, 0.025),
      hi = quantile(regret, 0.975),
      .groups = "drop"
    )
    ggplot(summ, aes(reorder(policy, mean_regret), mean_regret)) +
      geom_point(size = 3) +
      geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
      labs(x = "Policy", y = paste0("Regret after ", input$rep_horizon, " rounds")) +
      theme_minimal(base_size = 12)
  })
  output$compare_tbl <- renderDT({
    out <- comp_out()
    if (is.null(out)) return(datatable(tibble(Note = "Click Run Comparison")))
    summ <- out %>%
      group_by(policy) %>%
      summarize(
        n = dplyr::n(),
        mean = mean(regret),
        sd = sd(regret),
        median = median(regret),
        q025 = quantile(regret, 0.025),
        q975 = quantile(regret, 0.975),
        .groups = "drop"
      ) %>%
      mutate(across(c(mean, sd, median, q025, q975), ~round(.x, 3)))
    datatable(summ, rownames = FALSE, options = list(pageLength = 5))
  })
  output$compare_hint <- renderUI({
    if (is.null(comp_out())) {
      tags$p(em("No results yet — configure parameters and click ", strong("Run Comparison"), "."))
    } else { NULL }
  })
}

shinyApp(ui, server)
