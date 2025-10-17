# server

import socket
import time

HOST = 'localhost'
PORT = 6000

# Create TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[SERVER] Time server listening on {HOST}:{PORT}")

while True:
    conn, addr = server_socket.accept()
    print(f"[SERVER] Connection from {addr}")

    # Wait for client request
    data = conn.recv(1024).decode()
    if data == "TIME_REQUEST":
        server_time = time.time()
        conn.send(str(server_time).encode())
        print(f"[SERVER] Sent server time: {server_time}")
    conn.close()


# Client

import socket
import time

HOST = 'localhost'
PORT = 6000

# Step 1: Record send time (T1)
T1 = time.time()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
client_socket.send("TIME_REQUEST".encode())

# Step 2: Receive server time (T2)
server_time = float(client_socket.recv(1024).decode())
client_socket.close()

# Step 3: Record receive time (T3)
T3 = time.time()

# Step 4: Compute offset and adjusted time
round_trip_delay = (T3 - T1) / 2
adjusted_time = server_time + round_trip_delay

print(f"[CLIENT] Local send time (T1): {T1}")
print(f"[CLIENT] Server time (T2):     {server_time}")
print(f"[CLIENT] Local receive time (T3): {T3}")
print(f"[CLIENT] Estimated delay: {round_trip_delay:.6f} seconds")
print(f"[CLIENT] Synchronized time: {adjusted_time}")
