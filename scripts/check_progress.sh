#!/bin/bash

# Check evaluation progress

echo "=========================================="
echo "SAMSum Evaluation Progress"
echo "=========================================="
echo ""

# Check if evaluation is running
if docker compose ps | grep -q evaluator; then
    echo "Status: RUNNING ✓"
else
    echo "Status: NOT RUNNING"
fi

echo ""

# Check for SAMSum results
RUN_ID="${RUN_ID:-samsum_main}"
SAMSUM_RUN="results/samsum/run_${RUN_ID}"

if [ -d "$SAMSUM_RUN" ]; then
    echo "SAMSum Run: $RUN_ID"
    echo "---"
    
    if [ -f "$SAMSUM_RUN/outputs.jsonl" ]; then
        COMPLETED=$(wc -l < "$SAMSUM_RUN/outputs.jsonl" 2>/dev/null || echo 0)
        TOTAL=819
        PERCENT=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc 2>/dev/null || echo 0)
        
        echo "Examples processed: $COMPLETED / $TOTAL ($PERCENT%)"
        
        if [ $COMPLETED -eq $TOTAL ]; then
            echo "Status: ✅ COMPLETE"
        else
            REMAINING=$((TOTAL - COMPLETED))
            # Estimate time remaining (assume ~40s per example)
            TIME_REMAINING_MINS=$((REMAINING * 40 / 60))
            echo "Remaining: $REMAINING examples (~$TIME_REMAINING_MINS minutes)"
        fi
    else
        echo "Examples processed: 0 / 819 (0%)"
        echo "No outputs yet"
    fi
    
    echo ""
    
    if [ -f "$SAMSUM_RUN/summary.md" ]; then
        echo "✅ Evaluation complete!"
        echo ""
        echo "Quick stats:"
        grep -A 1 "ROUGE-L" "$SAMSUM_RUN/summary.md" | tail -1
        grep -A 1 "Consensus Similarity" "$SAMSUM_RUN/summary.md" | tail -1
        echo ""
        echo "Full results: cat $SAMSUM_RUN/summary.md"
    fi
else
    echo "No SAMSum run found at: $SAMSUM_RUN"
    echo "Start with: ./scripts/run_samsum_evaluation.sh"
fi

echo ""
echo "=========================================="
echo "To view full log: tail -f results/samsum/evaluation_log.txt"
echo "To stop evaluation: docker compose down"
echo "=========================================="

