;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Cliodynamics.jl
;; Media-Type: application/vnd.state+scm

(define-state Cliodynamics.jl
  (metadata
    (version "0.2.0")
    (schema-version "1.0.0")
    (created "2026-02-07")
    (updated "2026-02-13")
    (project "Cliodynamics.jl")
    (repo "hyperpolymath/Cliodynamics.jl"))

  (project-context
    (name "Cliodynamics.jl")
    (tagline "Julia package for quantitative modeling of historical dynamics and social complexity")
    (tech-stack ("Julia" "DifferentialEquations.jl" "DataFrames.jl" "Optim.jl" "Statistics" "LinearAlgebra")))

  (current-position
    (phase "beta")
    (overall-completion 100)
    (components
      ("malthusian-model" . 100)
      ("dst-model" . 100)
      ("elite-overproduction" . 100)
      ("political-stress" . 100)
      ("secular-cycles" . 100)
      ("state-capacity" . 100)
      ("collective-action" . 100)
      ("utility-functions" . 100)
      ("model-fitting" . 100)
      ("parameter-estimation" . 100)
      ("seshat-integration" . 100)
      ("plot-recipes" . 100)
      ("test-suite" . 100)
      ("documentation" . 95)
      ("examples" . 100)
      ("infrastructure" . 100))
    (working-features
      "Malthusian population dynamics model"
      "Demographic-structural theory (DST) model"
      "Elite overproduction index calculation"
      "Political stress indicator (PSI)"
      "Secular cycle analysis and phase detection"
      "State capacity model"
      "Collective action problem modeling"
      "Moving average utility function"
      "Detrending utility function"
      "Normalization utility function"
      "Carrying capacity calculation"
      "Crisis threshold detection"
      "Instability events identification"
      "Conflict intensity analysis"
      "Population pressure metrics"
      "Comprehensive test suite (16 exported functions)"
      "All tests passing"
      "Model fitting (Malthusian + Demographic-Structural)"
      "Parameter estimation with bootstrap confidence intervals"
      "Seshat Global History Databank integration"
      "Plots.jl recipes via package extension"
      "Polished examples with full analysis pipeline"))

  (route-to-mvp
    (milestones
      ((name "v0.1.0 - Core Models")
       (status "done")
       (completion 100)
       (items
         ("Malthusian model implementation" . done)
         ("DST model implementation" . done)
         ("Elite overproduction index" . done)
         ("Political stress indicator" . done)
         ("Secular cycle analysis" . done)
         ("State formation models" . done)
         ("Utility functions" . done)
         ("Comprehensive test suite" . done)))
      ((name "v0.2.0 - Examples & Data Integration")
       (status "done")
       (completion 100)
       (items
         ("Julia usage examples" . done)
         ("Historical dataset integration (Seshat)" . done)
         ("Plotting recipes for Plots.jl" . done)
         ("Model fitting to historical data" . done)
         ("Parameter estimation with Optim.jl" . done)))
      ((name "v1.0.0 - Production Release")
       (status "todo")
       (completion 0)
       (items
         ("Bayesian inference support (Turing.jl)" . todo)
         ("Spatial cliodynamic models" . todo)
         ("Interactive documentation (Documenter.jl)" . todo)
         ("Publication-quality examples" . todo)
         ("Julia General registry submission" . todo)))))

  (blockers-and-issues
    (critical ())
    (high ())
    (medium ())
    (low
      ("README.adoc duplicates README.md content")))

  (critical-next-actions
    (immediate
      "Write comprehensive Documenter.jl documentation")
    (this-week
      "Prepare for Julia General registry submission")
    (this-month
      "Bayesian parameter estimation with Turing.jl"
      "Spatial cliodynamic models"))

  (session-history
    ((date "2026-02-13")
     (actions
       "Completed v0.2.0: model fitting, parameter estimation, Seshat integration"
       "Added Plots.jl recipes via package extension (CliodynamicsPlotsExt)"
       "Rewrote examples with correct API and full analysis pipeline"
       "Fixed Seshat CSV parser to skip comment lines"
       "All 109 tests passing"))
    ((date "2026-02-12")
     (actions
       "Fixed SCM directory structure (.machines_readable/6scm -> .machine_readable)"
       "Updated all SPDX headers (AGPL -> PMPL)"
       "Completed template customization (ABI/FFI, K9, citations, AI manifest)"
       "Removed RSR template artifacts (RSR_OUTLINE.adoc)"
       "Fixed all Julia test failures: @kwdef structs, keyword args, sigmoid formula"
       "All 85 tests passing (was 49 pass, 3 fail, 5 error)"))))

;; Helper functions
(define (get-completion-percentage state)
  (current-position 'overall-completion state))

(define (get-blockers state severity)
  (blockers-and-issues severity state))

(define (get-milestone state name)
  (find (lambda (m) (equal? (car m) name))
        (route-to-mvp 'milestones state)))
