#!/bin/bash
echo "Graph-based propeties analysis"

echo "<<<<ssa2670-141.processed.cnf>>>>"
python eval/evaluate_graphs.py -d /Users/wenweihuang/Workspace/Net2SAT/result/generation/ssa2670-141.processed.cnf/

echo "<<<<mrpp_4x4#4_5.processed.cnf>>>>"
python eval/evaluate_graphs.py -d /Users/wenweihuang/Workspace/Net2SAT/result/generation/mrpp_4x4#4_5.processed.cnf/

echo "<<<<bf0432-007.processed.cnf>>>>"
python eval/evaluate_graphs.py -d /Users/wenweihuang/Workspace/Net2SAT/result/generation/bf0432-007.processed.cnf/

echo "<<<<bmc-ibm-7.processed.cnf>>>>"
python eval/evaluate_graphs.py -d /Users/wenweihuang/Workspace/Net2SAT/result/generation/bmc-ibm-7.processed.cnf/

echo "<<<<countbitsrotate016.processed.cnf>>>>"
python eval/evaluate_graphs.py -d /Users/wenweihuang/Workspace/Net2SAT/result/generation/countbitsrotate016.processed.cnf/

