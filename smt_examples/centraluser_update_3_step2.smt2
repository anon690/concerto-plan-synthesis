; SMT translation of the scheduling problem
; central-user architecture with 3 components
; attempting to schedule [update, install] on the components p0 and p1, and [suspend, deploy] on component u
; this problem is satisfiable, the model gives a schedule

(set-info :status sat)
(set-logic ALL)

(declare-datatypes ((behavior 0)) (((u_suspend_0) (u_deploy_1) (p0_update_0) (p0_install_1) (p1_update_0) (p1_install_1))))
(declare-datatypes ((step 0)) (((step0) (step1) (step2) (step3) (final_state))))

(declare-fun schedule (behavior) step)
(declare-fun succ (step) step)
(declare-fun toint (step) Int)
(declare-fun active_u_config (step) Bool)
(declare-fun active_u_service (step) Bool)
(declare-fun active_u_service0 (step) Bool)
(declare-fun active_u_config0 (step) Bool)
(declare-fun active_u_service1 (step) Bool)
(declare-fun active_u_config1 (step) Bool)
(declare-fun active_p0_config (step) Bool)
(declare-fun active_p0_service (step) Bool)
(declare-fun active_p1_config (step) Bool)
(declare-fun active_p1_service (step) Bool)

(assert (distinct (schedule u_suspend_0) final_state))
(assert (distinct (schedule u_deploy_1) final_state))
(assert (distinct (schedule p0_update_0) final_state))
(assert (distinct (schedule p0_install_1) final_state))
(assert (distinct (schedule p1_update_0) final_state))
(assert (distinct (schedule p1_install_1) final_state))
(assert (= (succ step0) step1))
(assert (= (succ step1) step2))
(assert (= (succ step2) step3))
(assert (= (succ step3) final_state))
(assert (= (succ final_state) final_state))
(assert (= (toint step0) 0))
(assert (= (toint step1) 1))
(assert (= (toint step2) 2))
(assert (= (toint step3) 3))
(assert (= (toint final_state) 4))
(assert (< (toint (schedule u_suspend_0)) (toint (schedule u_deploy_1))))
(assert (< (toint (schedule p0_update_0)) (toint (schedule p0_install_1))))
(assert (< (toint (schedule p1_update_0)) (toint (schedule p1_install_1))))
(assert (active_u_config step0))
(assert (active_u_service step0))
(assert (active_u_service0 step0))
(assert (active_u_config0 step0))
(assert (active_u_service1 step0))
(assert (active_u_config1 step0))
(assert (active_p0_config step0))
(assert (active_p0_service step0))
(assert (active_p1_config step0))
(assert (active_p1_service step0))
(assert (active_u_config (succ (schedule u_suspend_0))))
(assert (not (active_u_service (succ (schedule u_suspend_0)))))
(assert (not (active_u_service0 (succ (schedule u_suspend_0)))))
(assert (active_u_config0 (succ (schedule u_suspend_0))))
(assert (not (active_u_service1 (succ (schedule u_suspend_0)))))
(assert (active_u_config1 (succ (schedule u_suspend_0))))
(assert (active_u_config (succ (schedule u_deploy_1))))
(assert (active_u_service (succ (schedule u_deploy_1))))
(assert (active_u_service0 (succ (schedule u_deploy_1))))
(assert (active_u_config0 (succ (schedule u_deploy_1))))
(assert (active_u_service1 (succ (schedule u_deploy_1))))
(assert (active_u_config1 (succ (schedule u_deploy_1))))
(assert (active_p0_config (succ (schedule p0_update_0))))
(assert (not (active_p0_service (succ (schedule p0_update_0)))))
(assert (active_p0_config (succ (schedule p0_install_1))))
(assert (active_p0_service (succ (schedule p0_install_1))))
(assert (active_p1_config (succ (schedule p1_update_0))))
(assert (not (active_p1_service (succ (schedule p1_update_0)))))
(assert (active_p1_config (succ (schedule p1_install_1))))
(assert (active_p1_service (succ (schedule p1_install_1))))
(assert (active_p0_service (schedule u_deploy_1)))
(assert (distinct (schedule u_deploy_1) (schedule p0_update_0)))
(assert (active_p1_service (schedule u_deploy_1)))
(assert (distinct (schedule u_deploy_1) (schedule p1_update_0)))
(assert (not (active_u_service0 (schedule p0_update_0))))
(assert (distinct (schedule p0_update_0) (schedule u_deploy_1)))
(assert (not (active_u_service1 (schedule p1_update_0))))
(assert (distinct (schedule p1_update_0) (schedule u_deploy_1)))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_config step0) (active_u_config (succ step0)))))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_service step0) (active_u_service (succ step0)))))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_service0 step0) (active_u_service0 (succ step0)))))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_config0 step0) (active_u_config0 (succ step0)))))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_service1 step0) (active_u_service1 (succ step0)))))
(assert (=> (and (distinct step0 (schedule u_suspend_0)) (distinct step0 (schedule u_deploy_1))) (= (active_u_config1 step0) (active_u_config1 (succ step0)))))
(assert (=> (and (distinct step0 (schedule p0_update_0)) (distinct step0 (schedule p0_install_1))) (= (active_p0_config step0) (active_p0_config (succ step0)))))
(assert (=> (and (distinct step0 (schedule p0_update_0)) (distinct step0 (schedule p0_install_1))) (= (active_p0_service step0) (active_p0_service (succ step0)))))
(assert (=> (and (distinct step0 (schedule p1_update_0)) (distinct step0 (schedule p1_install_1))) (= (active_p1_config step0) (active_p1_config (succ step0)))))
(assert (=> (and (distinct step0 (schedule p1_update_0)) (distinct step0 (schedule p1_install_1))) (= (active_p1_service step0) (active_p1_service (succ step0)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_config step1) (active_u_config (succ step1)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_service step1) (active_u_service (succ step1)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_service0 step1) (active_u_service0 (succ step1)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_config0 step1) (active_u_config0 (succ step1)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_service1 step1) (active_u_service1 (succ step1)))))
(assert (=> (and (distinct step1 (schedule u_suspend_0)) (distinct step1 (schedule u_deploy_1))) (= (active_u_config1 step1) (active_u_config1 (succ step1)))))
(assert (=> (and (distinct step1 (schedule p0_update_0)) (distinct step1 (schedule p0_install_1))) (= (active_p0_config step1) (active_p0_config (succ step1)))))
(assert (=> (and (distinct step1 (schedule p0_update_0)) (distinct step1 (schedule p0_install_1))) (= (active_p0_service step1) (active_p0_service (succ step1)))))
(assert (=> (and (distinct step1 (schedule p1_update_0)) (distinct step1 (schedule p1_install_1))) (= (active_p1_config step1) (active_p1_config (succ step1)))))
(assert (=> (and (distinct step1 (schedule p1_update_0)) (distinct step1 (schedule p1_install_1))) (= (active_p1_service step1) (active_p1_service (succ step1)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_config step2) (active_u_config (succ step2)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_service step2) (active_u_service (succ step2)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_service0 step2) (active_u_service0 (succ step2)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_config0 step2) (active_u_config0 (succ step2)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_service1 step2) (active_u_service1 (succ step2)))))
(assert (=> (and (distinct step2 (schedule u_suspend_0)) (distinct step2 (schedule u_deploy_1))) (= (active_u_config1 step2) (active_u_config1 (succ step2)))))
(assert (=> (and (distinct step2 (schedule p0_update_0)) (distinct step2 (schedule p0_install_1))) (= (active_p0_config step2) (active_p0_config (succ step2)))))
(assert (=> (and (distinct step2 (schedule p0_update_0)) (distinct step2 (schedule p0_install_1))) (= (active_p0_service step2) (active_p0_service (succ step2)))))
(assert (=> (and (distinct step2 (schedule p1_update_0)) (distinct step2 (schedule p1_install_1))) (= (active_p1_config step2) (active_p1_config (succ step2)))))
(assert (=> (and (distinct step2 (schedule p1_update_0)) (distinct step2 (schedule p1_install_1))) (= (active_p1_service step2) (active_p1_service (succ step2)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_config step3) (active_u_config (succ step3)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_service step3) (active_u_service (succ step3)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_service0 step3) (active_u_service0 (succ step3)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_config0 step3) (active_u_config0 (succ step3)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_service1 step3) (active_u_service1 (succ step3)))))
(assert (=> (and (distinct step3 (schedule u_suspend_0)) (distinct step3 (schedule u_deploy_1))) (= (active_u_config1 step3) (active_u_config1 (succ step3)))))
(assert (=> (and (distinct step3 (schedule p0_update_0)) (distinct step3 (schedule p0_install_1))) (= (active_p0_config step3) (active_p0_config (succ step3)))))
(assert (=> (and (distinct step3 (schedule p0_update_0)) (distinct step3 (schedule p0_install_1))) (= (active_p0_service step3) (active_p0_service (succ step3)))))
(assert (=> (and (distinct step3 (schedule p1_update_0)) (distinct step3 (schedule p1_install_1))) (= (active_p1_config step3) (active_p1_config (succ step3)))))
(assert (=> (and (distinct step3 (schedule p1_update_0)) (distinct step3 (schedule p1_install_1))) (= (active_p1_service step3) (active_p1_service (succ step3)))))

(check-sat)
(get-model)