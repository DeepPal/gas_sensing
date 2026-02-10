@echo off
cd /d "c:\Users\deepp\Desktop\Codes\2_PYTHON_Multi_Sensing\gas_analysis"
python -m gas_analysis.tools.build_report "output\etoh_topavg" --title "Gas Analysis Report - EtOH with significance" --output-name "etoh_with_significance"
pause
