from concerto.assembly import ComponentType, Assembly


def parallel_user_type(n: int) -> ComponentType:
    t = ComponentType("parallel_user")
    pl_uninstalled = t.add_place("uninstalled")
    pl_allocated = t.add_place("allocated")
    pl_configured = t.add_place("configured")
    pl_running = t.add_place("running")
    t.add_provide_port("config", {pl_configured, pl_running})
    t.add_provide_port("service", {pl_running})
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_uninstalled, pl_allocated)
    bhv_deploy.add_transition("deploy4", pl_configured, pl_running)
    bhv_suspend = t.add_behavior("suspend")
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_configured, pl_uninstalled)
    for i in range(n):
        pl_sconf = t.add_place("sconf" + str(i))
        pl_suspended = t.add_place("suspended" + str(i))
        bhv_deploy.add_transition("deploy2" + str(i), pl_allocated, pl_sconf)
        bhv_deploy.add_transition("deploy3" + str(i), pl_sconf, pl_configured)
        bhv_suspend.add_transition("suspend1" + str(i), pl_running, pl_suspended)
        bhv_suspend.add_transition("suspend2" + str(i), pl_suspended, pl_configured)
        t.add_use_port("service" + str(i), {pl_running, pl_suspended})
        t.add_use_port("config" + str(i), {pl_sconf, pl_configured, pl_running, pl_suspended})
    return t


def provider_type() -> ComponentType:
    t = ComponentType("provider")
    pl_uninstalled = t.add_place("uninstalled")
    pl_installed = t.add_place("installed")
    pl_running = t.add_place("running")
    t.add_provide_port("config", {pl_installed, pl_running})
    t.add_provide_port("service", {pl_running})
    bhv_install = t.add_behavior("install")
    bhv_install.add_transition("install1", pl_uninstalled, pl_installed)
    bhv_install.add_transition("install2", pl_installed, pl_running)
    bhv_update = t.add_behavior("update")
    bhv_update.add_transition("update1", pl_running, pl_installed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_running, pl_uninstalled, 0)
    bhv_stop.add_transition("stop2", pl_installed, pl_uninstalled, 0)
    return t


def transformer_type() -> ComponentType:
    t = ComponentType("transformer")
    pl_uninstalled = t.add_place("uninstalled")
    pl_installed = t.add_place("installed")
    pl_configured = t.add_place("configured")
    pl_running = t.add_place("running")
    t.add_use_port("config_in", {pl_installed, pl_configured, pl_running})
    t.add_provide_port("config_out", {pl_configured, pl_running})
    t.add_use_port("service_in", {pl_running})
    t.add_provide_port("service_out", {pl_running})
    bhv_install = t.add_behavior("install")
    bhv_install.add_transition("install1", pl_uninstalled, pl_installed)
    bhv_install.add_transition("install2", pl_installed, pl_configured)
    bhv_install.add_transition("install3", pl_configured, pl_running)
    bhv_update = t.add_behavior("update")
    bhv_update.add_transition("update1", pl_running, pl_configured)
    bhv_suspend = t.add_behavior("suspend")
    bhv_suspend.add_transition("suspend1", pl_running, pl_installed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_running, pl_uninstalled, 0)
    bhv_stop.add_transition("stop2", pl_installed, pl_uninstalled, 1)
    bhv_stop.add_transition("stop3", pl_configured, pl_uninstalled, 2)
    return t


def centraluser_assembly(n: int) -> Assembly:
    assert n > 1
    a = Assembly()
    a.add_instance("u", parallel_user_type(n - 1))
    p_type = provider_type()
    for i in range(n - 1):
        p_id = "p" + str(i)
        a.add_instance(p_id, p_type)
        a.connect_instances_id(p_id, "config", "u", "config" + str(i))
        a.connect_instances_id(p_id, "service", "u", "service" + str(i))
    return a


def centralprovider_assembly(n: int) -> Assembly:
    assert n > 1
    a = Assembly()
    a.add_instance("p", provider_type())
    u_type = parallel_user_type(1)
    for i in range(n - 1):
        u_id = "u" + str(i)
        a.add_instance(u_id, u_type)
        a.connect_instances_id("p", "config", u_id, "config0")
        a.connect_instances_id("p", "service", u_id, "service0")
    return a


def linear_assembly(n: int) -> Assembly:
    assert n > 1
    a = Assembly()
    a.add_instance("p", provider_type())
    t_type = transformer_type()
    for i in range(n - 1):
        t_id = "t" + str(i)
        a.add_instance(t_id, t_type)
        if i == 0:
            a.connect_instances_id("p", "config", t_id, "config_in")
            a.connect_instances_id("p", "service", t_id, "service_in")
        else:
            a.connect_instances_id("t" + str(i - 1), "config_out", t_id, "config_in")
            a.connect_instances_id("t" + str(i - 1), "service_out", t_id, "service_in")
    return a


def stratified_assembly(n: int) -> Assembly:
    assert n > 1
    a = Assembly()
    n -= 2
    m = 1 if n == 0 else ((n - 1) % 3) + 1
    a.add_instance("u", parallel_user_type(m))
    a.add_instance("p", provider_type())
    if n == 0:
        a.connect_instances_id("p", "service", "u", "service0")
        a.connect_instances_id("p", "config", "u", "config0")
    for i in range(n):
        u_id = "u" + str(i)
        if i < 3:
            a.add_instance(u_id, parallel_user_type(1))
            a.connect_instances_id("p", "service", u_id, "service0")
            a.connect_instances_id("p", "config", u_id, "config0")
        else:
            a.add_instance(u_id, parallel_user_type(3))
            m = ((i - 3) // 3) * 3
            a.connect_instances_id("u" + str(m), "service", u_id, "service0")
            a.connect_instances_id("u" + str(m), "config", u_id, "config0")
            a.connect_instances_id("u" + str(m + 1), "service", u_id, "service1")
            a.connect_instances_id("u" + str(m + 1), "config", u_id, "config1")
            a.connect_instances_id("u" + str(m + 2), "service", u_id, "service2")
            a.connect_instances_id("u" + str(m + 2), "config", u_id, "config2")
        if i >= ((n - 1) // 3) * 3:
            a.connect_instances_id(u_id, "service", "u", "service" + str(i % 3))
            a.connect_instances_id(u_id, "config", "u", "config" + str(i % 3))
    return a
