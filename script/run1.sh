for proj in wss4j archiva deltaspike systemds lang; do
    python gitparse.py $proj >> log/log1.txt 2>> log/err1.txt
done