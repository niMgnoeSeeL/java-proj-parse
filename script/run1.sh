for proj in wss4j archiva deltaspike systemds ; do # lang을 뒤로 넘기겠다
    python gitparse.py $proj >> log/log1.txt 2>> log/err1.txt
done