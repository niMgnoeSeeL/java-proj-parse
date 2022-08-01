for proj in eagle bcel dbcp gora santuario; do
    python gitparse.py $proj >> log/log5.txt 2>> log/err5.txt
done