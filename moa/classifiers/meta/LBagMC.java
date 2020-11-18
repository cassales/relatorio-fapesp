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
import moa.capabilities.CapabilitiesHandler;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifierForkJoin;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

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
public class LBagMC extends AbstractClassifierForkJoin implements MultiClassClassifier,
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

    protected Classifier[] ensemble;

    protected ADWIN[] ADError;

    protected int numberOfChangesDetected;

    protected int[][] matrixCodes;

    protected boolean initMatrixCodes = false;

    protected boolean _Change = false;



    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        randomPoissonArray = new double[this.ensembleSizeOption.getValue()];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        this.ADError = new ADWIN[this.ensemble.length];
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ADError[i] = new ADWIN((double) this.deltaAdwinOption.getValue());
        }
        this.numberOfChangesDetected = 0;
        if (this.outputCodesOption.isSet()) {
            this.initMatrixCodes = true;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        int numClasses = inst.numClasses();
//        double t1 = System.currentTimeMillis();
//        _t1 = t1;
        //Output Codes
        if (this.initMatrixCodes) {
            this.matrixCodes = new int[this.ensemble.length][inst.numClasses()];
            for (int i = 0; i < this.ensemble.length; i++) {
                int numberOnes;
                int numberZeros;

                do { // until we have the same number of zeros and ones
                    numberOnes = 0;
                    numberZeros = 0;
                    for (int j = 0; j < numClasses; j++) {
                        int result = 0;
                        if (j == 1 && numClasses == 2) {
                            result = 1 - this.matrixCodes[i][0];
                        } else {
                            result = (this.classifierRandom.nextBoolean() ? 1 : 0);
                        }
                        this.matrixCodes[i][j] = result;
                        if (result == 1) {
                            numberOnes++;
                        } else {
                            numberZeros++;
                        }
                    }
                } while ((numberOnes - numberZeros) * (numberOnes - numberZeros) > (this.ensemble.length % 2));

            }
            this.initMatrixCodes = false;
        }
        boolean Change = false;
        _Change = false;
        Instance weightedInst = (Instance) inst.copy();
        double w = this.weightShrinkOption.getValue();
        int n = ensemble.length;
        if (_numOfCores != 1) {
            for (int i = 0; i < this.ensemble.length; i++) {
                double k = 0.0;
                switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                    case 0: //LBagMC
                        k = MiscUtils.poisson(w, this.classifierRandom);
                        break;
                    case 1: //LeveragingBagME
                        double error = this.ADError[i].getEstimation();
                        k = !this.ensemble[i].correctlyClassifies(weightedInst) ? 1.0 : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);
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
                randomPoissonArray[i] = k;
            }
////            BEGIN CASSALES
////            Stop counting time of this thread here. Time will be tracked individually on each call of the method train
////            We need to add time to cpuTime here in order to account for the for with the switch.
////            ADDENDUM
////            modifying to ThreadMXBean, which should capture time without adding time spent blocked.
////            double t2 = System.currentTimeMillis();
////            _cpuTime.addAndGet((int) (t2 - _t1));
////            If threadMXBean is not supported, fall back to previous implementation.
//            if (!ManagementFactory.getThreadMXBean().isCurrentThreadCpuTimeSupported()) {
//                double t2 = System.currentTimeMillis();
//                _cpuTime.addAndGet((int) (t2 - _t1));
//            }
////            END CASSALES
            if (_numOfCores == 0)
                IntStream.range(0, n).parallel().forEach(i -> train(i, inst));
            else {
                try { //Will create a Stream for each classifier on ensemble and try to execute them in parallel
                    //The problem is that threadpool will probably have less cores than the amount of threads created
                    //Leading to threads waiting to be executed with time counting.
                    _threadpool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> train(i, inst))).get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }
        } else { //Sequential code.
            //Train ensemble of classifiers
            for (int i = 0; i < this.ensemble.length; i++) {
                double k = 0.0;
                switch (this.leveraginBagAlgorithmOption.getChosenIndex()) {
                    case 0: //LBagMC
                        k = MiscUtils.poisson(w, this.classifierRandom);
                        break;
                    case 1: //LeveragingBagME
                        double error = this.ADError[i].getEstimation();
                        k = !this.ensemble[i].correctlyClassifies(weightedInst) ? 1.0 : (this.classifierRandom.nextDouble() < (error / (1.0 - error)) ? 1.0 : 0.0);
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
                if (k > 0) {
                    if (this.outputCodesOption.isSet()) {
                        weightedInst.setClassValue((double) this.matrixCodes[i][(int) inst.classValue()]);
                    }
                    weightedInst.setWeight(inst.weight() * k);
                    this.ensemble[i].trainOnInstance(weightedInst);
                }
                boolean correctlyClassifies = this.ensemble[i].correctlyClassifies(weightedInst);
                double ErrEstim = this.ADError[i].getEstimation();
                if (this.ADError[i].setInput(correctlyClassifies ? 0 : 1)) {
                    if (this.ADError[i].getEstimation() > ErrEstim) {
                        Change = true;
                    }
                }
            }
//            double t2 = System.currentTimeMillis();
//            _cpuTime.addAndGet((int) (t2 - _t1));
        }
//        BEGIN CASSALES
//        This was not being added to the count, is this correct?
//        END CASSALES
        if (Change || _Change) {
            //System.out.println("test");
            numberOfChangesDetected++;
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
                //this.ensemble[imax].trainOnInstance(inst);
                this.ADError[imax] = new ADWIN((double) this.deltaAdwinOption.getValue());
            }
        }
////        BEGIN CASSALES
////        Add a call to measure this thread time and add to cpu.
//        if (ManagementFactory.getThreadMXBean().isCurrentThreadCpuTimeSupported()) {
//            double nano = ManagementFactory.getThreadMXBean().getThreadCpuTime(Thread.currentThread().getId());
//            nano /= 1000000.0;
//            if (nano != -1)
//                _cpuTimeMXB.addAndGet((int) nano);
//        }
////        END CASSALES
    }

    @Override
    public void trainImpl(int index, Instance instance) {
////        BEGIN CASSALES
////        Start counting time again. This is done in order to avoid the time "waiting" for the thread to start
////        Since the submit method will create
////            ADDENDUM
////            modifying to ThreadMXBean, which should capture time without adding time spent blocked.
//        double t1 = System.currentTimeMillis();
////        END CASSALES
        double w = this.weightShrinkOption.getValue();
        Instance weightedInst = (Instance) instance.copy();
        double k = this.randomPoissonArray[index];


        if (k > 0) {
            if (this.outputCodesOption.isSet()) {
                weightedInst.setClassValue((double) this.matrixCodes[index][(int) instance.classValue()]);
            }
            weightedInst.setWeight(instance.weight() * k);
            this.ensemble[index].trainOnInstance(weightedInst);
        }
        boolean correctlyClassifies = this.ensemble[index].correctlyClassifies(weightedInst);
        double ErrEstim = this.ADError[index].getEstimation();
        if (this.ADError[index].setInput(correctlyClassifies ? 0 : 1)) {
            if (this.ADError[index].getEstimation() > ErrEstim) {
                _Change = true;
            }
        }
////            BEGIN CASSALES
////            modifying to ThreadMXBean, which should capture time without adding time spent blocked.
////        double t2 = System.currentTimeMillis();
////        _cpuTime.addAndGet((int) (t2 - t1));
//        double nano;
//        if (ManagementFactory.getThreadMXBean().isCurrentThreadCpuTimeSupported()) {
//            boolean a = ManagementFactory.getThreadMXBean().isThreadCpuTimeEnabled();
//            nano = ManagementFactory.getThreadMXBean().getCurrentThreadCpuTime();
//            nano /= 1000000.0;
//            double t2 = System.currentTimeMillis();
//            if (nano != -1)
//                _cpuTimeMXB.addAndGet((int) (nano));
//            _cpuTime.addAndGet((int) (t2 - t1));
//        } else {
//            double t2 = System.currentTimeMillis();
//            _cpuTime.addAndGet((int) (t2 - t1));
//        }
////        END CASSALES

    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
//        BEGIN CASSALES
//        I don't think this is being used.
//        double t1 = System.currentTimeMillis();
//        _t1 = t1;
//        END CASSALES

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
        if (this.getClass() == LBagMC.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

    @Override
    public void init() throws InterruptedException, ExecutionException {
    }
}

