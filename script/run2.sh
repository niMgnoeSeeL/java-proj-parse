for proj in lang net collections beanutils codec compress; do
    python gitparse.py $proj >> log/log2.txt 2>> log/err2.txt
done