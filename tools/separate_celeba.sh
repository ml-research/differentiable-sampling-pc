path=$1

path_val=$1/img_val
mkdir $path_val
path_test=$1/img_test
mkdir $path_test

cat $path/list_eval_partition.txt | while read line 
do
  # Get file name
  file=`echo $line | cut -d" " -f1`
  file_full=$path/img_align_celeba/$file
  ls $file_full

  # Get split type
  spl=`echo $line | cut -d" " -f2`
  echo $spl

  if [[ "$spl" == *"0"* ]]; then
    echo "train"
  fi

  if [[ "$spl" == *"1"* ]]; then
    echo "valid"
    cp $file_full $path_val/$file
  fi

  if [[ "$spl" == *"2"* ]]; then
    cp $file_full $path_test/$file
  fi

done