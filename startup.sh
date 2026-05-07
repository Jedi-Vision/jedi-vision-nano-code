#!/bin/bash

echo "Startup script ran at $(date)" >> /var/log/startup.log

sudo systemctl start bluetooth.service  # ensure bluetooth service is started (just in case)

cd jedi-vision-nano-code
bash start_container.sh