#!/bin/bash
function Y {
  #Usage: $0 FILE ALGORITHM RATE
  #alterar para memória do servidor
  Memory=300G
  CORES="0,1,2,3,4,5,6,7"
  #numactl --physcpubind=${CORES} java 
  #alterar caminhos
  #export MOA_HOME=/opt/data/moa-LAST
  export RESULT_DIR=/home/results/$3
  faux=${1##*\/}
  onlyname=${faux%%.*}
  echo "$2  $1 $3"
  if [[ $2 == *"MAX"* ]]; then
    IDENT="chunk"
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-100-8-50"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 100 -c 8) -s (ArffFileStream -f $1) -c 50 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-100-8-50" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-100-8-50
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-100-8-500"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 100 -c 8) -s (ArffFileStream -f $1) -c 500 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-100-8-500" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-100-8-500
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-100-8-2000"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 100 -c 8) -s (ArffFileStream -f $1) -c 2000 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-100-8-2000" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-100-8-2000

    # ENSEMBLE SIZE 150
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${1##*.}-150-8-50"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 150 -c 8) -s (ArffFileStream -f $1) -c 50 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-150-8-50" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-150-8-50
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-150-8-500"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 150 -c 8) -s (ArffFileStream -f $1) -c 500 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-150-8-500" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-150-8-500
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-150-8-2000"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EvaluateInterleavedTestThenTrainChunks -l ($2 -s 150 -c 8) -s (ArffFileStream -f $1) -c 2000 -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-150-8-2000" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-150-8-2000
  elif [[ ${2} == *"RUNPER"* ]]; then
    IDENT="interleaved"
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-100-8-1"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EITTTExperiments -l ($2 -s 100 -c 8) -s (ArffFileStream -f $1) -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-100-8-1" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-100-8-1
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-150-8-1"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EITTTExperiments -l ($2 -s 150 -c 8) -s (ArffFileStream -f $1) -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-150-8-1" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-150-8-1
  else
    IDENT="interleaved"
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-100-1-1"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EITTTExperiments -l ($2 -s 100) -s (ArffFileStream -f $1) -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-100-1-1" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-100-1-1
    echo "$RESULT_DIR/${IDENT}-${onlyname}-${2##*.}-150-1-1"
    numactl --physcpubind=${CORES} java -Xshare:off -XX:+UseParallelGC -Xmx$Memory -cp $MOA_HOME/lib/:$MOA_HOME/lib/moa.jar moa.DoTask "EITTTExperiments -l ($2 -s 150) -s (ArffFileStream -f $1) -e (BasicClassificationPerformanceEvaluator -o -p -r -f) -i -1 -d $RESULT_DIR/dump-${onlyname}-${2##*.}-150-1-1" > ${RESULT_DIR}/term-${IDENT}-${onlyname}-${2##*.}-150-1-1
  fi
  echo ""
}

function X {
  declare -a algs=(
  "meta.AdaptiveRandomForestSequential" "meta.AdaptiveRandomForestExecutorRUNPER" "meta.AdaptiveRandomForestExecutorMAXChunk"
  "meta.OzaBag" "meta.OzaBagExecutorRUNPER" "meta.OzaBagExecutorMAXChunk"
  "meta.OzaBagAdwin" "meta.OzaBagAdwinExecutorRUNPER" "meta.OzaBagAdwinExecutorMAXChunk"
  "meta.LeveragingBag" "meta.LBagExecutorRUNPER" "meta.LBagExecutorMAXChunk"
  "meta.OzaBagASHT" "meta.OzaBagASHTExecutorRUNPER" "meta.OzaBagASHTExecutorMAXChunk"
  "meta.StreamingRandomPatches" "meta.StreamingRandomPatchesExecutorRUNPER" "meta.StreamingRandomPatchesExecutorMAXChunk"
  )
  if [[ $2 == *"ARF"* ]]; then
    ID=0
  elif [[ $2 == "OBag" ]]; then
    ID=3
  elif [[ $2 == "OBagAd" ]]; then
    ID=6
  elif [[ $2 == "LBag" ]]; then
    ID=9
  elif [[ $2 == "OBagASHT" ]]; then
    ID=12
  elif [[ $2 == "SRP" ]]; then
    ID=15
  fi
  Y $1 ${algs[${ID}]} $3
  Y $1 ${algs[$(( ID+1 ))]} $3
  Y $1 ${algs[$(( ID+2 ))]} $3
}

# alterar para o caminho do HD/scratch
mkdir -p /home/results/first /home/results/second /home/results/third

#----------- FIRST ROUND
X $1elecNormNew.arff ARF first
X $1elecNormNew.arff LBag first
X $1elecNormNew.arff OBagAd first
X $1elecNormNew.arff OBag first
X $1elecNormNew.arff OBagASHT first
X $1elecNormNew.arff SRP first

X $1airlines.arff ARF first
X $1airlines.arff LBag first
X $1airlines.arff OBagAd first
X $1airlines.arff OBag first
X $1airlines.arff OBagASHT first
X $1airlines.arff SRP first

X $1covtypeNorm.arff ARF first
X $1covtypeNorm.arff LBag first
X $1covtypeNorm.arff OBagAd first
X $1covtypeNorm.arff OBag first
X $1covtypeNorm.arff OBagASHT first
X $1covtypeNorm.arff SRP first

X $1GMSC.arff ARF first
X $1GMSC.arff LBag first
X $1GMSC.arff OBagAd first
X $1GMSC.arff OBag first
X $1GMSC.arff OBagASHT first
X $1GMSC.arff SRP first


#----------- SECOND ROUND

X $1elecNormNew.arff ARF second
X $1elecNormNew.arff LBag second
X $1elecNormNew.arff OBagAd second
X $1elecNormNew.arff OBag second
X $1elecNormNew.arff OBagASHT second
X $1elecNormNew.arff SRP second

X $1airlines.arff ARF second
X $1airlines.arff LBag second
X $1airlines.arff OBagAd second
X $1airlines.arff OBag second
X $1airlines.arff OBagASHT second
X $1airlines.arff SRP second

X $1covtypeNorm.arff ARF second
X $1covtypeNorm.arff LBag second
X $1covtypeNorm.arff OBagAd second
X $1covtypeNorm.arff OBag second
X $1covtypeNorm.arff OBagASHT second
X $1covtypeNorm.arff SRP second

X $1GMSC.arff ARF second
X $1GMSC.arff LBag second
X $1GMSC.arff OBagAd second
X $1GMSC.arff OBag second
X $1GMSC.arff OBagASHT second
X $1GMSC.arff SRP second
#----------- THIRD ROUND

X $1elecNormNew.arff ARF third
X $1elecNormNew.arff LBag third
X $1elecNormNew.arff OBagAd third
X $1elecNormNew.arff OBag third
X $1elecNormNew.arff OBagASHT third
X $1elecNormNew.arff SRP third

X $1airlines.arff ARF third
X $1airlines.arff LBag third
X $1airlines.arff OBagAd third
X $1airlines.arff OBag third
X $1airlines.arff OBagASHT third
X $1airlines.arff SRP third

X $1covtypeNorm.arff ARF third
X $1covtypeNorm.arff LBag third
X $1covtypeNorm.arff OBagAd third
X $1covtypeNorm.arff OBag third
X $1covtypeNorm.arff OBagASHT third
X $1covtypeNorm.arff SRP third

X $1GMSC.arff ARF third
X $1GMSC.arff LBag third
X $1GMSC.arff OBagAd third
X $1GMSC.arff OBag third
X $1GMSC.arff OBagASHT third
X $1GMSC.arff SRP third
