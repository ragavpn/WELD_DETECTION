#!/bin/bash

# Check if directory path is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

DIRECTORY="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist"
    exit 1
fi

# Find and delete all non-image files in subdirectories
find "$DIRECTORY" -type f ! -name "*.jpg" ! -name "*.jpeg" ! -name "*.png" -delete -print

echo "Non-image files deletion completed"