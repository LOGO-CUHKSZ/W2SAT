#!/bin/bash
echo "Graph-based propeties analysis"
echo "<<<<generation instance>>>>"
python eval/evaluate_graphs.py -s eval/scalefree -d /Users/wenweihuang/Workspace/Net2SAT/eval_formulas/ssa2670-141/
# echo "<<<<Generating instance>>>>"
# python eval/evaluate_graphs.py -s /Users/wenweihuang/Workspace/Net2SAT/eval/scalefree -d /Users/wenweihuang/Workspace/Net2SAT/dataset/generating_formulas/ssa/