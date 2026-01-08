#!/bin/bash

# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET_DIR="/mnt/proj1/eu-25-40/innovaite/synchronized_data"
OUTPUT_FILE="folder_struct.txt"

# =============================================================================
# FUNCTIONS
# =============================================================================

analyze_structure() {
    echo "Root: $TARGET_DIR"
    echo "=========================================================="

    # 1. Check if root exists
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Directory $TARGET_DIR not found."
        return
    fi

    # 2. Iterate through Challenges (Level 1)
    # We sort to make output deterministic
    for challenge_path in $(find "$TARGET_DIR" -maxdepth 1 -mindepth 1 -type d | sort); do
        challenge_name=$(basename "$challenge_path")
        
        echo "  -- $challenge_name/"
        
        # --- A. Count Golden Truth (GT) ---
        # Look for 01_GT/SEG or 02_GT/SEG
        gt_count=0
        gt_folders=""
        
        for gt_ver in "01_GT" "02_GT"; do
            seg_path="$challenge_path/$gt_ver/SEG"
            if [ -d "$seg_path" ]; then
                # Count .tif files
                count=$(ls -1 "$seg_path"/*.tif 2>/dev/null | wc -l)
                gt_count=$((gt_count + count))
                if [ $count -gt 0 ]; then
                    gt_folders="$gt_folders $gt_ver($count)"
                fi
            fi
        done
        
        # --- B. Analyze Raters (Level 2) ---
        # We need to track consistency
        reference_tif_count=-1
        consistent_counts=true
        rater_count=0
        
        # Get list of subdirectories
        for item_path in $(find "$challenge_path" -maxdepth 1 -mindepth 1 -type d | sort); do
            item_name=$(basename "$item_path")
            
            # EXCLUSION LOGIC: Skip GT folders and raw data folders (01, 02)
            case "$item_name" in
                01_GT|02_GT|01|02) 
                    continue 
                    ;;
            esac
            
            rater_count=$((rater_count + 1))
            
            # --- C. Count TIFs in Rater's 01_RES folder ---
            res_path="$item_path/01_RES"
            if [ -d "$res_path" ]; then
                tif_count=$(ls -1 "$res_path"/*.tif 2>/dev/null | wc -l)
                
                # Check consistency
                if [ $reference_tif_count -eq -1 ]; then
                    reference_tif_count=$tif_count
                elif [ $tif_count -ne $reference_tif_count ]; then
                    consistent_counts=false
                fi
                
                # Draw the Rater node
                if [ $tif_count -eq $reference_tif_count ]; then
                    echo "      -- $item_name/ [OK: $tif_count tifs]"
                else
                    echo "      -- $item_name/ [WARNING: $tif_count tifs (Mismatch!)]"
                fi
            else
                echo "      -- $item_name/ [ERROR: No 01_RES folder]"
                consistent_counts=false
            fi
        done

        # --- D. Print Summary for this Challenge ---
        echo "      [SUMMARY $challenge_name]"
        echo "        > Raters found: $rater_count"
        
        if [ "$gt_folders" != "" ]; then
             echo "        > GT Files:     $gt_count Total $gt_folders"
        else
             echo "        > GT Files:     0 (No GT/SEG found)"
        fi

        if [ "$consistent_counts" = true ] && [ $rater_count -gt 0 ]; then
            echo "        > Consistency:  PASSED (All raters have $reference_tif_count files)"
        elif [ $rater_count -eq 0 ]; then
            echo "        > Consistency:  N/A (No raters found)"
        else
            echo "        > Consistency:  FAILED (File counts differ between raters!)"
        fi
        echo "" # Empty line for readability
    done
}

# =============================================================================
# EXECUTION
# =============================================================================

# Run the function and pipe output to both Terminal (tee) and File
echo "Starting analysis of $TARGET_DIR..."
analyze_structure | tee "$OUTPUT_FILE"
echo "Analysis complete. Saved to $OUTPUT_FILE"