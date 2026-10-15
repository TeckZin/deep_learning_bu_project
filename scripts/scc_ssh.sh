#!/bin/bash

set -a; source "$(dirname "$0")/../.env"; set +a
echo "user=[$BU_USER] pass length=${#BU_PASS}"

expect -c '
set timeout 30
spawn ssh $env(BU_USER)@scc1.bu.edu
expect {
  "yes/no"               { send "yes\r"; exp_continue }
  "assword:"             { send "$env(BU_PASS)\r"; exp_continue }
  -re "Passcode or option" { send "1\r" }
}
interact
'
