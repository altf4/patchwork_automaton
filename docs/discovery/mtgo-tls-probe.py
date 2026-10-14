"""One certificate-verified TLS handshake; no login or application messages."""
import socket, ssl, json, pathlib

host, port = 'mtgologin1.mtgo.com', 7770
context = ssl.create_default_context()
context.minimum_version = ssl.TLSVersion.TLSv1_2
result = {'host': host, 'port': port, 'application_bytes_sent': 0}
try:
    with socket.create_connection((host, port), timeout=10) as tcp:
        with context.wrap_socket(tcp, server_hostname=host) as tls:
            cert = tls.getpeercert()
            result.update(connected=True, tls_version=tls.version(), cipher=tls.cipher(),
                          certificate_subject=cert.get('subject'),
                          certificate_issuer=cert.get('issuer'),
                          certificate_names=cert.get('subjectAltName'),
                          certificate_expires=cert.get('notAfter'))
except Exception as error:
    result.update(connected=False, error_type=type(error).__name__, error=str(error))
path = pathlib.Path(__file__).with_name('tls_probe_result.json')
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
