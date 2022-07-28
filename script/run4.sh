for proj in scxml validator vfs giraph jspwiki; do
    python gitparse.py $proj >> log/log4.txt
done