HOST="10.10.1.1"
PORT=11211

# 连接 Telnet 服务器
(
  sleep 1
  echo "flush_all"
  echo "set serverNum 0 0 1"
  echo "0"
  echo "set clientNum 0 0 1"
  echo "0"
  echo "quit"
) | telnet $HOST $PORT