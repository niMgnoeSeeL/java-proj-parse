for proj in net collections beanutils codec compress; do
    python gitparse.py $proj >> log/log2.txt
done