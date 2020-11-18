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
import moa.classifiers.AbstractClassifierForkJoin;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

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
public class OzaBagMC extends AbstractClassifierForkJoin implements MultiClassClassifier {

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


    protected Classifier[] _classifiers;
    protected Instance _instance;
    protected Random _r;
    protected int[] _weight;


    public void resetLearningImpl() {

        _r = new Random(_randomSeedOption.getValue());
        int ensembleSize = _ensembleSizeOption.getValue();
        Classifier baseLearner = (Classifier) getPreparedClassOption(_baseLearnerOption);
        baseLearner.resetLearning();
        _classifiers = new Classifier[ensembleSize];
        for (int i = 0; i < ensembleSize; i++) {
            _classifiers[i] = (Classifier) baseLearner.copy();
        }
        _weight = new int[ensembleSize];
    }


    public void trainOnInstanceImpl(Instance inst) {
        int n = _classifiers.length;
        for (int i = 0; i < n; i++)
            _weight[i] = MiscUtils.poisson(1.0, _r);
        if (_numOfCores == 0) {
            IntStream.range(0, n).parallel().forEach(i -> train(i, inst));
        } else if (_numOfCores == 1) {
            for (int i = 0; i < n; i++)
                train(i, inst);
        } else {
            try {
//                .get() makes it and overall blocking call
                _threadpool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> train(i, inst))).get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    public void trainOnInstanceImpl(ArrayList<Instance> instances) {
        for (Instance inst : instances) {
            int n = _classifiers.length;
            for (int i = 0; i < n; i++)
                _weight[i] = MiscUtils.poisson(1.0, _r);
            if (_numOfCores == 0) {
                IntStream.range(0, n).parallel().forEach(i -> train(i, inst));
            } else if (_numOfCores == 1) {
                for (int i = 0; i < n; i++)
                    train(i, inst);
            } else {
                try {
                    _threadpool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> train(i, inst))).get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }
        }
    }


    public void trainImpl(int index, Instance instance) {
        int k = _weight[index];
        if (k > 0) {
            Instance weightedInst = (Instance) instance.copy();
            weightedInst.setWeight(instance.weight() * k);
            _classifiers[index].trainOnInstance(weightedInst);
        }
    }

    //Initial Method Of algorithm incase developers want to use it.
    public void init() throws InterruptedException, ExecutionException {

    }


    public double[] getVotesForInstance(Instance instance) {
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this._classifiers.length; i++) {
            DoubleVector vote = new DoubleVector(this._classifiers[i].getVotesForInstance(instance));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    // Avoids Thread Pool Leaking
    public void trainingHasEnded() {
        if (_threadpool != null)
            _threadpool.shutdown();

    }

    public boolean isRandomizable() {
        return true;
    }

    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size", _classifiers == null ? 0 : _classifiers.length)};
    }

    public Classifier[] getSubClassifiers() {
        return Arrays.copyOf(_classifiers, _classifiers.length);
    }
}