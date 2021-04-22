from concerto.assembly import ComponentType, Assembly
from concerto.scheduling import ReconfigurationProblem, SchedulingStatistics


def mariadb_type() -> ComponentType:
    t = ComponentType("mariadb")
    pl_initiated = t.add_place("initiated")
    pl_configured = t.add_place("configured")
    pl_bootstrapped = t.add_place("bootstrapped")
    pl_restarted = t.add_place("restarted")
    pl_registered = t.add_place("registered")
    pl_deployed = t.add_place("deployed")
    pl_interrupted = t.add_place("interrupted")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy11", pl_initiated, pl_configured)
    bhv_deploy.add_transition("deploy12", pl_initiated, pl_configured)
    bhv_deploy.add_transition("deploy2", pl_configured, pl_bootstrapped)
    bhv_deploy.add_transition("deploy3", pl_bootstrapped, pl_restarted)
    bhv_deploy.add_transition("deploy4", pl_restarted, pl_registered)
    bhv_deploy.add_transition("deploy5", pl_registered, pl_deployed)
    bhv_interrupt = t.add_behavior("interrupt")
    bhv_interrupt.add_transition("interrupt1", pl_deployed, pl_interrupted)
    bhv_pause = t.add_behavior("pause")
    bhv_pause.add_transition("pause1", pl_interrupted, pl_bootstrapped)
    bhv_stop = t.add_behavior("update")
    bhv_stop.add_transition("update1", pl_interrupted, pl_configured)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_interrupted, pl_initiated)
    t.add_use_port("haproxy_service", {pl_bootstrapped, pl_restarted})
    t.add_use_port("common_service", {pl_restarted, pl_registered, pl_deployed, pl_interrupted})
    t.add_provide_port("service", {pl_deployed})
    return t


def keystone_type() -> ComponentType:
    t = ComponentType("keystone")
    pl_initiated = t.add_place("initiated")
    pl_pulled = t.add_place("pulled")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy2", pl_pulled, pl_deployed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_deployed, pl_pulled)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("mariadb_service", {pl_pulled, pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def nova_type() -> ComponentType:
    t = ComponentType("nova")
    pl_initiated = t.add_place("initiated")
    pl_pulled = t.add_place("pulled")
    pl_ready = t.add_place("ready")
    pl_restarted = t.add_place("restarted")
    pl_deployed = t.add_place("deployed")
    pl_interrupted = t.add_place("interrupted")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy11", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy12", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy13", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy21", pl_pulled, pl_ready)
    bhv_deploy.add_transition("deploy22", pl_pulled, pl_ready)
    bhv_deploy.add_transition("deploy3", pl_ready, pl_restarted)
    bhv_deploy.add_transition("deploy4", pl_restarted, pl_deployed)
    bhv_deploy.add_transition("deploy5", pl_initiated, pl_deployed)
    bhv_interrupt = t.add_behavior("interrupt")
    bhv_interrupt.add_transition("interrupt1", pl_deployed, pl_interrupted)
    bhv_pause = t.add_behavior("pause")
    bhv_pause.add_transition("pause1", pl_interrupted, pl_ready)
    bhv_stop = t.add_behavior("update")
    bhv_stop.add_transition("update1", pl_interrupted, pl_pulled)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_interrupted, pl_initiated)
    t.add_use_port("mariadb_service", {pl_pulled, pl_ready, pl_restarted, pl_deployed})
    t.add_use_port("keystone_service", {pl_ready, pl_restarted, pl_deployed, pl_interrupted})
    t.add_provide_port("service", {pl_deployed})
    return t


def neutron_type() -> ComponentType:
    t = ComponentType("neutron")
    pl_initiated = t.add_place("initiated")
    pl_pulled = t.add_place("pulled")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy11", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy12", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy13", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy2", pl_pulled, pl_deployed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_deployed, pl_pulled)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("mariadb_service", {pl_pulled, pl_deployed})
    t.add_use_port("keystone_service", {pl_pulled, pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def glance_type() -> ComponentType:
    t = ComponentType("glance")
    pl_initiated = t.add_place("initiated")
    pl_pulled = t.add_place("pulled")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy11", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy12", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy13", pl_initiated, pl_pulled)
    bhv_deploy.add_transition("deploy2", pl_pulled, pl_deployed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_deployed, pl_pulled)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("mariadb_service", {pl_pulled, pl_deployed})
    t.add_use_port("keystone_service", {pl_pulled, pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def facts_type() -> ComponentType:
    t = ComponentType("facts")
    pl_initiated = t.add_place("initiated")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_deployed)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_provide_port("service", {pl_deployed})
    return t


def haproxy_type() -> ComponentType:
    t = ComponentType("haproxy")
    pl_initiated = t.add_place("initiated")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_deployed)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("facts_service", {pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def memcached_type() -> ComponentType:
    t = ComponentType("memcached")
    pl_initiated = t.add_place("initiated")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_deployed)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("facts_service", {pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def ovswitch_type() -> ComponentType:
    t = ComponentType("ovswitch")
    pl_initiated = t.add_place("initiated")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_deployed)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("facts_service", {pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def rabbitmq_type() -> ComponentType:
    t = ComponentType("rabbitmq")
    pl_initiated = t.add_place("initiated")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_deployed)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("facts_service", {pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


def common_type() -> ComponentType:
    t = ComponentType("common")
    pl_initiated = t.add_place("initiated")
    pl_configured = t.add_place("configured")
    pl_deployed = t.add_place("deployed")
    bhv_deploy = t.add_behavior("deploy")
    bhv_deploy.add_transition("deploy1", pl_initiated, pl_configured)
    bhv_deploy.add_transition("deploy2", pl_configured, pl_deployed)
    bhv_stop = t.add_behavior("stop")
    bhv_stop.add_transition("stop1", pl_deployed, pl_configured)
    bhv_uninstall = t.add_behavior("uninstall")
    bhv_uninstall.add_transition("uninstall1", pl_deployed, pl_initiated)
    t.add_use_port("facts_service", {pl_configured, pl_deployed})
    t.add_provide_port("service", {pl_deployed})
    return t


if __name__ == "__main__":
    a = Assembly()
    a.add_instance("facts", facts_type())
    a.add_instance("haproxy", haproxy_type())
    a.add_instance("common", common_type())
    a.add_instance("rabbitmq", rabbitmq_type())
    a.add_instance("memcached", memcached_type())
    a.add_instance("ovswitch", ovswitch_type())
    m = a.add_instance("mariadb", mariadb_type())
    k = a.add_instance("keystone", keystone_type())
    no = a.add_instance("nova", nova_type())
    ne = a.add_instance("neutron", neutron_type())
    g = a.add_instance("glance", glance_type())
    a.connect_instances_id("facts", "service", "haproxy", "facts_service")
    a.connect_instances_id("facts", "service", "common", "facts_service")
    a.connect_instances_id("facts", "service", "rabbitmq", "facts_service")
    a.connect_instances_id("facts", "service", "memcached", "facts_service")
    a.connect_instances_id("facts", "service", "ovswitch", "facts_service")
    a.connect_instances_id("haproxy", "service", "mariadb", "haproxy_service")
    a.connect_instances_id("common", "service", "mariadb", "common_service")
    a.connect_instances_id("mariadb", "service", "keystone", "mariadb_service")
    a.connect_instances_id("mariadb", "service", "nova", "mariadb_service")
    a.connect_instances_id("mariadb", "service", "neutron", "mariadb_service")
    a.connect_instances_id("mariadb", "service", "glance", "mariadb_service")
    a.connect_instances_id("keystone", "service", "nova", "keystone_service")
    a.connect_instances_id("keystone", "service", "neutron", "keystone_service")
    a.connect_instances_id("keystone", "service", "glance", "keystone_service")

    initially_active_places = {c: {c.type().get_place("deployed")} for c in a.instances()}
    goal_behaviors = {m: {m.type().get_behavior("update"), m.type().get_behavior("deploy")}}
    goal_state = {}  # no need to specify since all use ports must be active, like in this initial state
    p = ReconfigurationProblem(a,
                               initially_active_places,
                               goal_behaviors,
                               goal_state,
                               ReconfigurationProblem.sortkey)
    solution = p.solve(verbose=True)
    SchedulingStatistics.output()
    print("solution\n")
    print(solution.script_string(explicit_waitall=False))
