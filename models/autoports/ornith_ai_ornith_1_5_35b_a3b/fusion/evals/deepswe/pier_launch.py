"""Run the Pier CLI (datacurve-pier 0.3.1) with one local-plumbing patch.

Pier puts agent containers of no-internet tasks on an internal docker network whose only way out
is a squid sidecar that allows the agent's inference host on ports 80/443 only. Our model endpoint
is the llm_proxy on the docker bridge gateway at a high port, so the port is added to squid's
Safe_ports (PIER_EXTRA_SAFE_PORTS, comma-separated). The host allowlist itself is unchanged (Pier
derives it from ANTHROPIC_BASE_URL: the gateway IP only), so the agent still reaches nothing else.

usage: pier_launch.py <pier args...>     (e.g. run -c job.json -y | job resume -p <job dir>)
"""
import os
import sys

import pier.environments.agent_setup as agent_setup

_orig = agent_setup.squid_bootstrap_command
PORTS = [p for p in os.environ.get("PIER_EXTRA_SAFE_PORTS", "").split(",") if p.strip()]


def _patched():
    s = _orig()
    if PORTS:
        extra = "".join(f"acl Safe_ports port {int(p)}\n" for p in PORTS)
        s = s.replace("acl Safe_ports port 80 443\n", "acl Safe_ports port 80 443\n" + extra)
        assert extra in s, "pier squid config changed; update pier_launch.py"
    return s


agent_setup.squid_bootstrap_command = _patched

if __name__ == "__main__":
    from pier.cli.main import app

    sys.argv = ["pier"] + sys.argv[1:]
    app()
