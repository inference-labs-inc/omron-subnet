import os
import time
from OpenSSL import crypto
import bittensor as bt


class CertificateManager:
    def __init__(self, cert_path: str):
        self.cert_path = cert_path
        self.key_path = os.path.join(cert_path, "key.pem")
        self.cert_file = os.path.join(cert_path, "cert.pem")

    def ensure_valid_certificate(self, external_ip: str) -> None:
        if not os.path.exists(self.cert_file):
            bt.logging.warning(
                "Certificate not found. Generating new self-signed certificate."
            )
            self._generate_certificate(external_ip)

    def _generate_certificate(self, cn: str) -> None:
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 4096)

        cert = crypto.X509()
        cert.get_subject().CN = cn
        cert.set_serial_number(int(time.time()))
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(2 * 365 * 24 * 60 * 60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")

        os.makedirs(os.path.dirname(self.cert_file), exist_ok=True)
        with open(self.cert_file, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        with open(self.key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))