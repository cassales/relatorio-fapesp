/*
 *    LBagMC.java
 *    Copyright (C) 2010 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
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

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifierExecutorServiceChunk;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import weka.core.pmml.jaxbbindings.False;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

/**
 * Leveraging Bagging for evolving data streams using ADWIN. Leveraging Bagging
 * and Leveraging Bagging MC using Random Output Codes ( -o option).
 *
 * <p>See details in:<br /> Albert Bifet, Geoffrey Holmes, Bernhard Pfahringer.
 * Leveraging Bagging for Evolving Data Streams Machine Learning and Knowledge
 * Discovery in Databases, European Conference, ECML PKDD}, 2010.</p>
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */
public class LBagExecutorCHUNK extends AbstractClassifierExecutorServiceChunk implements MultiClassClassifier,
        CapabilitiesHandler {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Leveraging Bagging for evolving data streams using ADWIN.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");


    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);

    public FloatOption weightShrinkOption = new FloatOption("weightShrink", 'w',
            "The number to use to compute the weight of new instances.", 6, 0.0, Float.MAX_VALUE);

    public FloatOption deltaAdwinOption = new FloatOption("deltaAdwin", 'a',
            "Delta of Adwin change detection", 0.002, 0.0, 1.0);

    // Leveraging Bagging MC: uses this option to use Output Codes
    public FlagOption outputCodesOption = new FlagOption("outputCodes", 'o',
            "Use Output Codes to use binary classifiers.");

    public MultiChoiceOption leveraginBagAlgorithmOption = new MultiChoiceOption(
            "leveraginBagAlgorithm", 'm', "Leveraging Bagging to use.", new String[]{
            "LBagMC", "LeveragingBagME", "LeveragingBagHalf", "LeveragingBagWT", "LeveragingSubag"},
            new String[]{"Leveraging Bagging for evolving data streams using ADWIN",
                    "Leveraging Bagging ME using weight 1 if misclassified, otherwise error/(1-error)",
                    "Leveraging Bagging Half using resampling without replacement half of the instances",
                    "Leveraging Bagging WT without taking out all instances.",
                    "Leveraging Subagging using resampling without replacement."
            }, 0);

    public FlagOption _parallelOption = new FlagOption("parallel", 'p',
            "Run ensemble in parallel.");

    protected double[] randomPoissonArray;

    protected HoeffdingTree[] ensemble;

    protected ADWIN[] ADError;

    protected int numberOfChangesDetected;

    protected int[][] matrixCodes;

    protected boolean initMatrixCodes = false;

    protected boolean _Change = false;

    protected ArrayList<TrainingRunnable> trainers;

    protected int batchesProcessed;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new HoeffdingTree[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        randomPoissonArray = new double[this.ensembleSizeOption.getValue()];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = (HoeffdingTree) baseLearner.copy();
        }
        this.ADError = new ADWIN[this.ensemble.length];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ADError[i] = new ADWIN((double) this.deltaAdwinOption.getValue());
        }
        this.numberOfChangesDetected = 0;
        this.batchesProcessed = 0;
        if (this.outputCodesOption.isSet()) {
            this.initMatrixCodes = true;
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
            boolean oco = this.outputCodesOption.isSet();
            this.trainers = new ArrayList<>();
            for (int i = 0; i < this.ensemble.length; i++) {
                TrainingRunnable trainer = new TrainingRunnable(this.ensemble[i], this.ADError[i], oco);
                trainers.add(trainer);
            }
        }
    }

    @Override
    public void trainOnInstances(Instances instances) {
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance inst = instances.instance(i).copy();
            int numClasses = inst.numClasses();
            if (this.initMatrixCodes) {
                this.matrixCodes = new int[this.ensemble.length][inst.numClasses()];
                for (int j = 0; j < this.ensemble.length; j++) {
                    int numberOnes;
                    int numberZeros;

                    do { // until we have the same number of zeros and ones
                        numberOnes = 0;
                        numberZeros = 0;
                        for (int k = 0; k < numClasses; k++) {
                            int result = 0;
                            if (k == 1 && numClasses == 2) {
                                result = 1 - this.matrixCodes[j][0];
                            } else {
                                result = (this.classifierRandom.nextBoolean() ? 1 : 0);
                            }
                            this.matrixCodes[j][k] = result;
                            if (result == 1) {
                                numberOnes++;
                            } else {
                                numberZeros++;
                            }
                        }
                    } while ((numberOnes - numberZeros) * (numberOnes - numberZeros) > (this.ensemble.length % 2));
                }
            }

            for (TrainingRunnable t : this.trainers) {
                double w = this.weightShrinkOption.getValue();
                double k = 0.0;
                switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                    case 0: //LBagMC
                        k = MiscUtils.poisson(w, this.classifierRandom);
                        break;
                    case 1: //LeveragingBagME
                        double error = this.ADError[i].getEstimation();
                        k = !this.ensemble[i].correctlyClassifies(instances.instance(i).copy()) ?
                                1.0 : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);
                        break;
                    case 2: //LeveragingBagHalf
                        w = 1.0;
                        k = this.classifierRandom.nextBoolean() ? 0.0 : w;
                        break;
                    case 3: //LeveragingBagWT
                        w = 1.0;
                        k = 1.0 + MiscUtils.poisson(w, this.classifierRandom);
                        break;
                    case 4: //LeveragingSubag
                        w = 1.0;
                        k = MiscUtils.poisson(1, this.classifierRandom);
                        k = (k > 0) ? w : 0;
                        break;
                }
                t.weights.add(k);
            }
        }
        for (TrainingRunnable l : this.trainers)
            l.instances = new Instances(instances);
        this.batchesProcessed++;
        if (this._threadpool != null) {
            try {
                this._threadpool.invokeAll(this.trainers);
            } catch (InterruptedException ex) {
                throw new RuntimeException("Could not call invokeAll() on training threads.");
            }
        }
        if (_Change) {
            _Change = false;
            numberOfChangesDetected++;
            System.out.println("Change # " + numberOfChangesDetected + " detected on batch # " + this.batchesProcessed);
            double max = 0.0;
            int imax = -1;
            for (int i = 0; i < this.ensemble.length; i++) {
                if (max < this.ADError[i].getEstimation()) {
                    max = this.ADError[i].getEstimation();
                    imax = i;
                }
            }
            if (imax != -1) {
                this.ensemble[imax].resetLearning();
                this.ADError[imax] = new ADWIN((double) this.deltaAdwinOption.getValue());
                this.trainers.get(imax).ADError = this.ADError[imax];
            }
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.outputCodesOption.isSet()) {
            return getVotesForInstanceBinary(inst);
        }
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    public double[] getVotesForInstanceBinary(Instance inst) {
        double combinedVote[] = new double[(int) inst.numClasses()];
        Instance weightedInst = (Instance) inst.copy();
        if (this.initMatrixCodes == false) {
            for (int i = 0; i < this.ensemble.length; i++) {
                //Replace class by OC
                weightedInst.setClassValue((double) this.matrixCodes[i][(int) inst.classValue()]);

                double vote[];
                vote = this.ensemble[i].getVotesForInstance(weightedInst);
                //Binary Case
                int voteClass = 0;
                if (vote.length == 2) {
                    voteClass = (vote[1] > vote[0] ? 1 : 0);
                }
                //Update votes
                for (int j = 0; j < inst.numClasses(); j++) {
                    if (this.matrixCodes[i][j] == voteClass) {
                        combinedVote[j] += 1;
                    }
                }
            }
        }
        return combinedVote;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                this.ensemble != null ? this.ensemble.length : 0),
                new Measurement("change detections", this.numberOfChangesDetected)
        };
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == LBagExecutorCHUNK.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    @Override
    public void init() throws InterruptedException, ExecutionException {
    }


    /***
     * Inner class to assist with the multi-thread execution.
     */
    protected class TrainingRunnable implements Runnable, Callable<Integer> {
        private Classifier learner;
        private Instances instances;
        private ArrayList<Double> weights;
        protected ADWIN ADError;
        protected boolean outputCodesOptionIsSet;
        protected int[] matrixCodes;

        public TrainingRunnable(Classifier learner, ADWIN ADError, boolean ocos) {
            this.learner = learner;
            this.ADError = ADError;
            this.weights = new ArrayList<>();
            this.outputCodesOptionIsSet = ocos;
        }

        @Override
        public void run() {
            for (int i = 0; i < this.instances.numInstances(); i++) {
                Instance weightedInst = this.instances.instance(i).copy();
                if (this.outputCodesOptionIsSet) {
                    weightedInst.setClassValue((double) this.matrixCodes[(int) weightedInst.classValue()]);
                }
                weightedInst.setWeight(this.instances.instance(i).weight() * this.weights.get(i));
                this.learner.trainOnInstance(weightedInst);
                boolean correctlyClassifies = this.learner.correctlyClassifies(weightedInst);
                double ErrEstim = this.ADError.getEstimation();
                if (this.ADError.setInput(correctlyClassifies ? 0 : 1)) {
                    if (this.ADError.getEstimation() > ErrEstim) {
                        _Change = true;
                    }
                }
            }
            this.weights.clear();
        }

        @Override
        public Integer call() {
            run();
            return 0;
        }
    }
}

