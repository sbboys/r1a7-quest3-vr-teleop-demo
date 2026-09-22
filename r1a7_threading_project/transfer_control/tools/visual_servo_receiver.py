import socket
import json


class VisualServoReceiver:

    def __init__(self, port=5005):

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.sock.bind(
            ("127.0.0.1", port)
        )

        self.sock.setblocking(False)

        self.delta = [0.0,0.0,0.0]


    def update(self):

        try:
            data,_ = self.sock.recvfrom(1024)

            msg=json.loads(
                data.decode()
            )

            self.delta[0]=float(
                msg.get("dx_mm",0)
            )

            self.delta[1]=float(
                msg.get("dy_mm",0)
            )

            self.delta[2]=float(
                msg.get("dz_mm",0)
            )

        except BlockingIOError:
            pass

        return self.delta
