for proj in configuration digester jcs imaging io; do 
    python gitparse.py $proj >> log/log3.txt  2>> log/err3.txt
done