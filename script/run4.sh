for proj in scxml validator vfs giraph jspwiki; do
    python gitparse.py $proj >> log/log4.txt 2>> log/err4.txt
done