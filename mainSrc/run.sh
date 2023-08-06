#!/bin/bash

#relitive path to AIP
PathToAIP="AIP"

(sh src/runAIP.sh $PathToAIP &) | tee /dev/tty | grep -m 1 "Quit the server with CONTROL-C."
echo "done with loading AIP"
sh src/runAIP.sh