#!/bin/bash

# Script: auto_upload_to_drive.sh
# Purpose: Wait for packing to finish and then upload to Google Drive.

TARGET_FILE="/mnt/18TData/facediff/data/ovoxel_cache.tar"
PACKING_LOG="/mnt/18TData/facediff/data/packing_progress.log"
REMOTE_NAME="facediffgdrive"
REMOTE_FOLDER="FaceDiff Data"

echo "Starting Automation Process at $(date)"

# 1. Wait for the tar process (PID) to finish
# We check if the process we started (tar -cSvf) is still alive
while ps -p 1464977 > /dev/null; do
    echo "Still packing... Current size: $(du -sh $TARGET_FILE | cut -f1). Waiting 2 minutes..."
    sleep 120
done

echo "Packing FINISHED at $(date)!"
echo "Final Size: $(du -sh $TARGET_FILE)"

# 2. Upload to Google Drive using rclone
echo "Starting Upload to Google Drive..."
rclone copy -P "$TARGET_FILE" "$REMOTE_NAME":"$REMOTE_FOLDER/"

if [ $? -eq 0 ]; then
    echo "===================================================="
    echo "SUCCESS: Everything uploaded to Google Drive!"
    echo "Time: $(date)"
    echo "===================================================="
else
    echo "ERROR: Upload failed! Please check your internet or Drive space."
fi
