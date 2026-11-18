"""Keep Python network traffic in the test process and its children local."""

from functools import wraps
import ipaddress
import os
import socket


WORKSPACE_ENV = "WQU_OFFLINE_TEST_WORKSPACE"
_CREDENTIALS = (
    "BRAIN_USERNAME", "BRAIN_PASSWORD", "OPENALEX_API_KEY",
    "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALLOW_AGENT_SUBMIT",
)


class RemoteNetworkBlocked(OSError):
    """A test attempted to use a remote network endpoint."""


def _check_host(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if host in (None, "", "localhost", "localhost."):
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise RemoteNetworkBlocked("pytest permits only loopback network traffic; mock the remote server")


def _check_address(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and address is not None:
        _check_host(address[0])


def _guard_connect(method):
    @wraps(method)
    def connect(sock, address, /):
        # Check the hostname before socket.connect can look it up in C.
        _check_address(sock, address)
        return method(sock, address)

    return connect


def _guard_lookup(method, *, sockaddr=False):
    @wraps(method)
    def lookup(host, *args, **kwargs):
        _check_host(host[0] if sockaddr else host)
        return method(host, *args, **kwargs)

    return lookup


def _guard_sendto(method):
    @wraps(method)
    def sendto(sock, data, *args):
        if args:
            _check_address(sock, args[-1])
        return method(sock, data, *args)

    return sendto


def _guard_sendmsg(method):
    @wraps(method)
    def sendmsg(sock, buffers, ancdata=(), flags=0, address=None, /):
        _check_address(sock, address)
        return method(sock, buffers, ancdata, flags, address)

    return sendmsg


def install(set_attr):
    """Guard the socket APIs shared by requests and urllib, preserving loopback."""
    for name in ("connect", "connect_ex"):
        set_attr(socket.socket, name, _guard_connect(getattr(socket.socket, name)))
    set_attr(socket.socket, "sendto", _guard_sendto(socket.socket.sendto))
    if hasattr(socket.socket, "sendmsg"):
        set_attr(socket.socket, "sendmsg", _guard_sendmsg(socket.socket.sendmsg))
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        set_attr(socket, name, _guard_lookup(getattr(socket, name)))
    set_attr(socket, "getnameinfo", _guard_lookup(socket.getnameinfo, sockaddr=True))


def isolate_environment(set_env, workspace, *, child=False):
    if not child or not os.environ.get("BRAIN_RESEARCH_DIR"):
        set_env("BRAIN_RESEARCH_DIR", workspace)
    for key in _CREDENTIALS:
        # Empty values also prevent the repo's .env loader from filling them.
        if not child or key not in os.environ:
            set_env(key, "")
