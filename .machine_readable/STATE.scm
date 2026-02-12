;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Project state tracking for Cliodynamics.jl
;; Media-Type: application/vnd.state+scm

(define-state Cliodynamics.jl
  (metadata
    (version "0.1.0")
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
    (overall-completion 95)
    (components
      ("malthusian-model" . 100)
      ("dst-model" . 100)
      ("elite-overproduction" . 100)
      ("political-stress" . 100)
      ("secular-cycles" . 100)
      ("state-capacity" . 100)
      ("collective-action" . 100)
      ("utility-functions" . 100)
      ("test-suite" . 100)
      ("documentation" . 90)
      ("examples" . 60)
      ("infrastructure" . 95))
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
      "All tests passing"))

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
       (status "in-progress")
       (completion 60)
       (items
         ("Julia usage examples" . in-progress)
         ("Historical dataset integration (Seshat)" . todo)
         ("Plotting recipes for Plots.jl" . todo)
         ("Model fitting to historical data" . todo)
         ("Parameter estimation with Optim.jl" . todo)))
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
    (medium
      ("Documentation needs polish for registry submission"))
    (low
      ("Plots dependency removed - examples need updating"
       "README.adoc duplicates README.md content")))

  (critical-next-actions
    (immediate
      "Complete Julia usage examples (basic_usage.jl, historical_analysis.jl)"
      "Polish documentation for registry submission")
    (this-week
      "Integrate with Seshat Global History Databank"
      "Add plotting recipes for model outputs"
      "Write comprehensive Documenter.jl documentation")
    (this-month
      "Model fitting utilities with Optim.jl"
      "Bayesian parameter estimation with Turing.jl"
      "Prepare for Julia General registry submission"))

  (session-history
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
