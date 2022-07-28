for proj in wss4j archiva deltaspike systemds lang; do
    python gitparse.py $proj >> log/log1.txt
done