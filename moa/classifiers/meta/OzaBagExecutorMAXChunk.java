/*
 *    OzaBagBMod1.java
 *    Copyright (C) 2019 University of Waikato, Hamilton, New Zealand
 *    @author Bernhard Pfahringr (bernhard@waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */


package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifierExecutorServiceChunk;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;

/**
 * Incremental on-line bagging of Oza and Russell.
 *
 * <p>Oza and Russell developed online versions of bagging and boosting for
 * Data Streams. They show how the process of sampling bootstrap replicates
 * from training data can be simulated in a data stream context. They observe
 * that the probability that any individual example will be chosen for a
 * replicate tends to a Poisson(1) distribution.</p>
 *
 * <p>[OR] N. Oza and S. Russell. Online bagging and boosting.
 * In Artiﬁcial Intelligence and Statistics 2001, pages 105–112.
 * Morgan Kaufmann, 2001.</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Classifier to train</li>
 * <li>-n : The ensemble size</li>
 * <li>-p : Run in parallel</li>
 * <li>-s : The random seed</li> </ul>
 *
 * @author Bernhard Pfahringer (bernhard@waikato.ac.nz)
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public class OzaBagExecutorMAXChunk extends AbstractClassifierExecutorServiceChunk implements MultiClassClassifier {

    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }

    private static final long serialVersionUID = 2L;

    public ClassOption _baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption _ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The ensemble size.", 10, 1, Integer.MAX_VALUE);

    public IntOption _randomSeedOption = new IntOption("randomSeed", 'r',
            "The random seed.", 42, -Integer.MAX_VALUE, Integer.MAX_VALUE);

    protected HoeffdingTree[] ensemble;
    protected Random _r;

    protected ArrayList<OzaBagExecutorMAXChunk.TrainingRunnable> trainers;

    public void resetLearningImpl() {
        _r = new Random(_randomSeedOption.getValue());
        int ensembleSize = _ensembleSizeOption.getValue();
        Classifier baseLearner = (Classifier) getPreparedClassOption(_baseLearnerOption);
        baseLearner.resetLearning();
        ensemble = new HoeffdingTree[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) {
            ensemble[i] = (HoeffdingTree) baseLearner.copy();
        }

        // Multi-threading
        int numberOfJobs;
        if (this._amountOfCores.getValue() == -1)
            numberOfJobs = Runtime.getRuntime().availableProcessors();
        else
            numberOfJobs = this._amountOfCores.getValue();
        // SINGLE_THREAD and requesting for only 1 thread are equivalent.
        // this.executor will be null and not used...
        if (numberOfJobs != 0 && numberOfJobs != 1) {
            this._threadpool = Executors.newFixedThreadPool(numberOfJobs);
            this.trainers = new ArrayList<>();
            for (int i = 0; i < this.ensemble.length; i++) {
                TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i]);
                trainers.add(trainer);
            }
        }
    }

    @Override
    public void trainOnInstances(Instances instances) {
        for (TrainingRunnable l : trainers)
            l.insts = new Instances(instances);
        if (this._threadpool != null) {
            try {
                this._threadpool.invokeAll(trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
    }

    //Initial Method Of algorithm incase developers want to use it.
    public void init() throws InterruptedException, ExecutionException {

    }


    public double[] getVotesForInstance(Instance instance) {
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(instance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    public boolean isRandomizable() {
        return true;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size", ensemble == null ? 0 : ensemble.length)};
    }

    public Classifier[] getSubClassifiers() {
        return Arrays.copyOf(ensemble, ensemble.length);
    }

    /***
     * Inner class to assist with the multi-thread execution.
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        private Classifier learner;
        private Instances insts;


        public TrainingRunnable(Classifier learner) {
            this.learner = learner;
            this.insts = new Instances();
        }

        @Override
        public void run() {
            for (int i = 0; i < this.insts.numInstances(); i++) {
                int k = MiscUtils.poisson(1.0, ThreadLocalRandom.current());
                Instance weightedInst = this.insts.instance(i);
                weightedInst.setWeight(weightedInst.weight() * k);
                this.learner.trainOnInstance(weightedInst);
            }
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }
    }
}

