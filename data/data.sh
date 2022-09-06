# download WMT20 test data
mkdir -p test/test20
cd test/test20
for lang_pair in en-de en-zh ro-en et-en ne-en si-en
do
  wget https://github.com/facebookresearch/mlqe/raw/main/data/${lang_pair}_test.tar.gz
  tar -zxvf ${lang_pair}_test.tar.gz
  rm_hyphen=`echo "$lang_pair" | sed -e "s/-//"`
  cp ${lang_pair}/test20.${rm_hyphen}.df.short.tsv .
  rm -r $lang_pair ${lang_pair}_test.tar.gz
done
cd ../../
python wmt20.py

# download WMT21 test data
mkdir -p test/test21
cd test/test21
for lang_pair in en-de en-zh ro-en et-en ne-en si-en
do
  wget https://github.com/sheffieldnlp/mlqe-pe/raw/master/data/test21_goldlabels/${lang_pair}-test21.tar.gz
  tar -zxvf ${lang_pair}-test21.tar.gz
  mv ${lang_pair}-test21/goldlabels/test21.da ${lang_pair}-test21
  rm -r ${lang_pair}-test21/goldlabels      \
        ${lang_pair}-test21/word-probas     \
        ${lang_pair}-test21/test21.tok.mt   \
        ${lang_pair}-test21/test21.tok.src  \
        ${lang_pair}-test21.tar.gz
done
cd ../../

# download train data
mkdir -p train
cd train
for lang_pair in en-de en-zh ro-en et-en ne-en si-en
do
    wget https://www.quest.dcs.shef.ac.uk/wmt20_files_qe/training_${lang_pair}.tar.gz
    tar -zxvf training_${lang_pair}.tar.gz
    rm training_${lang_pair}.tar.gz
done

# detokenize train data
cd ../../tools_external
for lang_pair in ende enzh roen eten neen sien
do
    src_lang=`echo $lang_pair | sed -e "s/.\{2\}$//"`
    trg_lang=`echo $lang_pair | sed -e "s/^.\{2\}//"`
    ./detokenizer.perl -l $src_lang < ../data/train/train.${lang_pair}.${src_lang} > ../data/train/train.${lang_pair}.${src_lang}.detok
    ./detokenizer.perl -l $trg_lang < ../data/train/train.${lang_pair}.${trg_lang} > ../data/train/train.${lang_pair}.${trg_lang}.detok
done
cd ../data

# embedding
python embedding.py
