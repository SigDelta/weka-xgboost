package weka.classifiers.trees;

import ml.dmlc.xgboost4j.java.Booster;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.xgboost.DMatrixLoader;
import weka.classifiers.trees.xgboost.OptionWithType;
import weka.core.*;
import ml.dmlc.xgboost4j.java.DMatrix;
import weka.core.Capabilities.Capability;

import java.io.Serializable;
import java.util.*;


/**
 * <!-- globalinfo-start -->
 * * Class for a XGBoost classifier.
 * * <br><br>
 * <!-- globalinfo-end -->
 * <p>
 * <!-- technical-bibtex-start -->
 * * BibTeX:
 * * <pre>
 * * &#64;misc{missing_id
 * * }
 * * </pre>
 * * <br><br>
 * <!-- technical-bibtex-end -->
 * <p>
 * <!-- options-start -->
 * * Valid options are: <p>
 * *
 * * <pre> -force-probability-distribution
 * * Force probability distribution</pre>
 * *
 * * <pre> -booster &lt;string&gt;
 * * [default=gbtree]
 * * which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function</pre>
 * *
 * * <pre> -silent &lt;integer&gt;
 * * [default=0]
 * * 0 means printing running messages, 1 means silent mode.</pre>
 * *
 * * <pre> -nthread &lt;integer&gt;
 * * [default to maximum number of threads available if not set]
 * * number of parallel threads used to run xgboost</pre>
 * *
 * * <pre> -num_pbuffer &lt;integer&gt;
 * * [set automatically by xgboost, no need to be set by user]
 * * size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step</pre>
 * *
 * * <pre> -num_feature &lt;integer&gt;
 * * [set automatically by xgboost, no need to be set by user]
 * * feature dimension used in boosting, set to maximum dimension of the feature</pre>
 * *
 * * <pre> -eta &lt;double&gt;
 * * [default=0.3, alias: learning_rate], range: [0,1]
 * * step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.</pre>
 * *
 * * <pre> -learning_rate &lt;double&gt;
 * * [default=0.3, alias: eta]. </pre>
 * *
 * * <pre> -gamma &lt;double&gt;
 * * [default=0, alias: min_split_loss], range: [0,∞]
 * * minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.</pre>
 * *
 * * <pre> -min_split_loss &lt;double&gt;
 * * alias: gamma</pre>
 * *
 * * <pre> -max_depth &lt;integer&gt;
 * * [default=6], range: [1,∞]
 * * maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.</pre>
 * *
 * * <pre> -min_child_weight &lt;double&gt;
 * * [default=1], range: [0,∞]
 * * minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.</pre>
 * *
 * * <pre> -max_delta_step &lt;double&gt;
 * * [default=0], range: [0,∞]
 * * Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update</pre>
 * *
 * * <pre> -subsample &lt;double&gt;
 * *  [default=1], range: (0,1],
 * * subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.</pre>
 * *
 * * <pre> -colsample_bytree &lt;double&gt;
 * * [default=1], range: (0,1]
 * * subsample ratio of columns when constructing each tree.</pre>
 * *
 * * <pre> -colsample_bylevel &lt;double&gt;
 * * [default=1], range: (0,1]
 * * subsample ratio of columns for each split, in each level.</pre>
 * *
 * * <pre> -lambda &lt;double&gt;
 * * [default=1, alias: reg_lambda]
 * * L2 regularization term on weights, increase this value will make model more conservative.</pre>
 * *
 * * <pre> -reg_lambda &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -alpha &lt;double&gt;
 * * [default=0, alias: reg_alpha]
 * * L1 regularization term on weights, increase this value will make model more conservative.</pre>
 * *
 * * <pre> -reg_alpha &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -tree_method &lt;string&gt;
 * * [default='auto'], The tree construction algorithm used in XGBoost; Choices: {'auto', 'exact', 'approx'} </pre>
 * *
 * * <pre> -sketch_eps &lt;double&gt;
 * * [default=0.03], range: (0, 1)
 * * Only used for approximate greedy algorithm. This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy. Usually user does not have to tune this. but consider setting to a lower number for more accurate enumeration.</pre>
 * *
 * * <pre> -scale_pos_weight &lt;double&gt;
 * * [default=1]
 * * Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases)</pre>
 * *
 * * <pre> -updater &lt;string&gt;
 * * [default='grow_colmaker,prune']
 * * A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters.</pre>
 * *
 * * <pre> -refresh_leaf &lt;integer&gt;
 * * [default=1]
 * * This is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as tree nodes' stats are updated. When it is false, only node stats are updated.</pre>
 * *
 * * <pre> -process_type &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -sample_type &lt;string&gt;
 * * [default="uniform"]
 * * type of sampling algorithm:
 * * -"uniform": dropped trees are selected uniformly.
 * * -"weighted": dropped trees are selected in proportion to weight.</pre>
 * *
 * * <pre> -normalize_type &lt;string&gt;
 * * [default="tree"]
 * * type of normalization algorithm:
 * * -"tree": new trees have the same weight of each of dropped trees.
 * * -"forest": new trees have the same weight of sum of dropped trees (forest).</pre>
 * *
 * * <pre> -rate_drop &lt;double&gt;
 * * [default=0.0], range: [0.0, 1.0]
 * * dropout rate (a fraction of previous trees to drop during the dropout)</pre>
 * *
 * * <pre> -one_drop &lt;integer&gt;
 * * [default=0]
 * * when this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).</pre>
 * *
 * * <pre> -skip_drop &lt;double&gt;
 * * [default=0.0], range: [0.0, 1.0]
 * * Probability of skipping the dropout procedure during a boosting iteration.</pre>
 * *
 * * <pre> -lambda_bias &lt;double&gt;
 * * [default=0, alias: reg_lambda_bias]
 * * L2 regularization term on bias (no L1 reg on bias because it is not important)</pre>
 * *
 * * <pre> -reg_lambda_bias &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -objective &lt;string&gt;
 * * [ default=reg:linear ]
 * * "reg:linear" --linear regression
 * * "reg:logistic" --logistic regression
 * * "binary:logistic" --logistic regression for binary classification, output probability
 * * "binary:logitraw" --logistic regression for binary classification, output score before logistic transformation
 * * "count:poisson" --poisson regression for count data, output mean of poisson distribution [max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)]
 * * "multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
 * * "multi:softprob" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
 * * "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
 * * "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed
 * * "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.</pre>
 * *
 * * <pre> -num_class &lt;integer&gt;
 * * number of classes</pre>
 * *
 * * <pre> -base_score &lt;double&gt;
 * * [ default=0.5 ]
 * * -the initial prediction score of all instances, global bias
 * * -for sufficient number of iterations, changing this value will not have too much effect.</pre>
 * *
 * * <pre> -eval_metric &lt;string&gt;
 * * [ default according to objective ]
 * * evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )</pre>
 * *
 * * <pre> -seed &lt;integer&gt;
 * * [ default=0 ]
 * * random number seed.</pre>
 * *
 * * <pre> -tweedie_variance_power &lt;double&gt;
 * * [default=1.5], range: (1,2)
 * * parameter that controls the variance of the Tweedie distribution
 * * -set closer to 2 to shift towards a gamma distribution
 * * -set closer to 1 to shift towards a Poisson distribution.</pre>
 * *
 * * <pre> -output-debug-info
 * *  If set, classifier is run in debug mode and
 * *  may output additional info to the console</pre>
 * *
 * * <pre> -do-not-check-capabilities
 * *  If set, classifier capabilities are not checked before classifier is built
 * *  (use with caution).</pre>
 * *
 * * <pre> -num-decimal-places
 * *  The number of decimal places for the output of numbers in the model (default 2).</pre>
 * *
 * * <pre> -batch-size
 * *  The desired batch size for batch prediction  (default 100).</pre>
 * *
 * <!-- options-end -->
 *
 * @author Michal Wasiluk (michal@wasiluk.io)
 */
public class XGBoost extends AbstractClassifier implements OptionHandler, TechnicalInformationHandler, Serializable {

    private static final long serialVersionUID = 1141447363965993342L;

    private Booster booster;

    Map<String, Object> params = new HashMap<>();

    static List<OptionWithType> xgBoostParamsOptions = new ArrayList<>();

    static Set<String> probabilityObjective = new HashSet<>(Arrays.asList("binary:logistic", "multi:softprob"));

    private boolean forceProbabilityDistribution = false;

    static {
        // General Parameters
        addStringOption("booster", "[default=gbtree]\nwhich booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function");
        addIntOption("silent", "[default=0]\n0 means printing running messages, 1 means silent mode.");
        addIntOption("nthread", "[default to maximum number of threads available if not set]\nnumber of parallel threads used to run xgboost");
        addIntOption("num_pbuffer", "[set automatically by xgboost, no need to be set by user]\nsize of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step");
        addIntOption("num_feature", "[set automatically by xgboost, no need to be set by user]\nfeature dimension used in boosting, set to maximum dimension of the feature");

        // Parameters for Tree Booster
        addDoubleOption("eta", "[default=0.3, alias: learning_rate], range: [0,1]\nstep size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.");
        addDoubleOption("learning_rate", "[default=0.3, alias: eta]. ");
        addDoubleOption("gamma", "[default=0, alias: min_split_loss], range: [0,∞]\nminimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.");
        addDoubleOption("min_split_loss", "alias: gamma");
        addIntOption("max_depth", "[default=6], range: [1,∞]\nmaximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.");
        addDoubleOption("min_child_weight", "[default=1], range: [0,∞]\nminimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.");
        addDoubleOption("max_delta_step", "[default=0], range: [0,∞]\nMaximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update");
        addDoubleOption("subsample", " [default=1], range: (0,1], \n" +
                "subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.");
        addDoubleOption("colsample_bytree", "[default=1], range: (0,1]\nsubsample ratio of columns when constructing each tree.");
        addDoubleOption("colsample_bylevel", "[default=1], range: (0,1]\nsubsample ratio of columns for each split, in each level.");
        addDoubleOption("lambda", "[default=1, alias: reg_lambda]\nL2 regularization term on weights, increase this value will make model more conservative.");
        addDoubleOption("reg_lambda");
        addDoubleOption("alpha", "[default=0, alias: reg_alpha]\nL1 regularization term on weights, increase this value will make model more conservative.");
        addDoubleOption("reg_alpha");
        addStringOption("tree_method", "[default='auto'], The tree construction algorithm used in XGBoost; Choices: {'auto', 'exact', 'approx'} ");
        addDoubleOption("sketch_eps", "[default=0.03], range: (0, 1)\nOnly used for approximate greedy algorithm. This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy. Usually user does not have to tune this. but consider setting to a lower number for more accurate enumeration.");
        addDoubleOption("scale_pos_weight", "[default=1]\nControl the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases)");
        addStringOption("updater", "[default='grow_colmaker,prune']\nA comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters.");
        addIntOption("refresh_leaf", "[default=1]\nThis is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as tree nodes' stats are updated. When it is false, only node stats are updated.");
        addStringOption("process_type");

        // Additional parameters for Dart Booster
        addStringOption("sample_type", "[default=\"uniform\"]\ntype of sampling algorithm:\n-\"uniform\": dropped trees are selected uniformly.\n-\"weighted\": dropped trees are selected in proportion to weight.");
        addStringOption("normalize_type", "[default=\"tree\"]\ntype of normalization algorithm:\n-\"tree\": new trees have the same weight of each of dropped trees.\n-\"forest\": new trees have the same weight of sum of dropped trees (forest).");
        addDoubleOption("rate_drop", "[default=0.0], range: [0.0, 1.0]\ndropout rate (a fraction of previous trees to drop during the dropout)");
        addIntOption("one_drop", "[default=0]\nwhen this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).");
        addDoubleOption("skip_drop", "[default=0.0], range: [0.0, 1.0]\nProbability of skipping the dropout procedure during a boosting iteration.");

        //Additional parameters for Linear Booster
//        addDoubleOption("lambda");
//        addDoubleOption("alpha");
        addDoubleOption("lambda_bias", "[default=0, alias: reg_lambda_bias]\nL2 regularization term on bias (no L1 reg on bias because it is not important)");
        addDoubleOption("reg_lambda_bias");

        //Learning Task Parameters
        addStringOption("objective", "[ default=reg:linear ]\n" +
                "\"reg:linear\" --linear regression\n" +
                "\"reg:logistic\" --logistic regression\n" +
                "\"binary:logistic\" --logistic regression for binary classification, output probability\n" +
                "\"binary:logitraw\" --logistic regression for binary classification, output score before logistic transformation\n" +
                "\"count:poisson\" --poisson regression for count data, output mean of poisson distribution [max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)]\n" +
                "\"multi:softmax\" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)\n" +
                "\"multi:softprob\" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.\n" +
                "\"rank:pairwise\" --set XGBoost to do ranking task by minimizing the pairwise loss\n" +
                "\"reg:gamma\" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed\n" +
                "\"reg:tweedie\" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.");
        addIntOption("num_class", "number of classes");
        addDoubleOption("base_score", "[ default=0.5 ]\n-the initial prediction score of all instances, global bias\n-for sufficient number of iterations, changing this value will not have too much effect.");
        addStringOption("eval_metric", "[ default according to objective ]\nevaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )");
        addIntOption("seed", "[ default=0 ]\nrandom number seed.");

        //Parameters for Tweedie Regression
        addDoubleOption("tweedie_variance_power", "[default=1.5], range: (1,2)\nparameter that controls the variance of the Tweedie distribution\n-set closer to 2 to shift towards a gamma distribution\n-set closer to 1 to shift towards a Poisson distribution.");

    }

    private Integer numRound = 20; // number of iterations


    public String globalInfo() {
        return "Class for a XGBoost classifier.";
    }

    public void buildClassifier(Instances instances) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        DMatrix dmat = DMatrixLoader.instancesToDMatrix(instances);

        Map<String, DMatrix> watches = new HashMap<>();
        watches.put("train", dmat);

        if (!params.containsKey("num_class")) {
            params.put("num_class", instances.numClasses());
        }

        booster = ml.dmlc.xgboost4j.java.XGBoost.train(dmat, params, numRound, watches, null, null);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (!forceProbabilityDistribution && !probabilityObjective.contains(params.get("objective"))) {
            return super.distributionForInstance(instance);
        }

        DMatrix dmat = DMatrixLoader.instanceToDenseDMatrix(instance);
        float[][] predict1 = booster.predict(dmat);
        float[] predict = predict1[0];

        double[] predictDouble = new double[predict.length];
        for (int i = 0; i < predict.length; i++) {
//            predictDouble[i] = Double.valueOf(String.valueOf(predict[i]));
            predictDouble[i] = predict[i];
        }

        return predictDouble;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        if (forceProbabilityDistribution && probabilityObjective.contains(params.get("objective"))) {
            return super.classifyInstance(instance);
        }
        DMatrix dmat = DMatrixLoader.instanceToDMatrix(instance);
        float[][] predict1 = booster.predict(dmat);
        float[] predict = predict1[0];

        double[] predictDouble = new double[predict.length];
        for (int i = 0; i < predict.length; i++) {
//            predictDouble[i] = Double.valueOf(String.valueOf(predict[i]));
            predictDouble[i] = predict[i];
        }

        return predictDouble[0];

    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(3);

        newVector.addElement(new Option("Force probability distribution", "force-probability-distribution", 0, "-force-probability-distribution"));

        xgBoostParamsOptions.forEach(newVector::addElement);

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    }

    /**
     * Parses a given list of options.
     * <p/>
     * <p>
     * <!-- options-start -->
     * * Valid options are: <p>
     * *
     * * <pre> -force-probability-distribution
     * * Force probability distribution</pre>
     * *
     * * <pre> -booster &lt;string&gt;
     * * [default=gbtree]
     * * which booster to use, can be gbtree, gblinear or dart. gbtree and dart use tree based model while gblinear uses linear function</pre>
     * *
     * * <pre> -silent &lt;integer&gt;
     * * [default=0]
     * * 0 means printing running messages, 1 means silent mode.</pre>
     * *
     * * <pre> -nthread &lt;integer&gt;
     * * [default to maximum number of threads available if not set]
     * * number of parallel threads used to run xgboost</pre>
     * *
     * * <pre> -num_pbuffer &lt;integer&gt;
     * * [set automatically by xgboost, no need to be set by user]
     * * size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step</pre>
     * *
     * * <pre> -num_feature &lt;integer&gt;
     * * [set automatically by xgboost, no need to be set by user]
     * * feature dimension used in boosting, set to maximum dimension of the feature</pre>
     * *
     * * <pre> -eta &lt;double&gt;
     * * [default=0.3, alias: learning_rate], range: [0,1]
     * * step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinks the feature weights to make the boosting process more conservative.</pre>
     * *
     * * <pre> -learning_rate &lt;double&gt;
     * * [default=0.3, alias: eta]. </pre>
     * *
     * * <pre> -gamma &lt;double&gt;
     * * [default=0, alias: min_split_loss], range: [0,∞]
     * * minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be.</pre>
     * *
     * * <pre> -min_split_loss &lt;double&gt;
     * * alias: gamma</pre>
     * *
     * * <pre> -max_depth &lt;integer&gt;
     * * [default=6], range: [1,∞]
     * * maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.</pre>
     * *
     * * <pre> -min_child_weight &lt;double&gt;
     * * [default=1], range: [0,∞]
     * * minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be.</pre>
     * *
     * * <pre> -max_delta_step &lt;double&gt;
     * * [default=0], range: [0,∞]
     * * Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update</pre>
     * *
     * * <pre> -subsample &lt;double&gt;
     * *  [default=1], range: (0,1],
     * * subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.</pre>
     * *
     * * <pre> -colsample_bytree &lt;double&gt;
     * * [default=1], range: (0,1]
     * * subsample ratio of columns when constructing each tree.</pre>
     * *
     * * <pre> -colsample_bylevel &lt;double&gt;
     * * [default=1], range: (0,1]
     * * subsample ratio of columns for each split, in each level.</pre>
     * *
     * * <pre> -lambda &lt;double&gt;
     * * [default=1, alias: reg_lambda]
     * * L2 regularization term on weights, increase this value will make model more conservative.</pre>
     * *
     * * <pre> -reg_lambda &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -alpha &lt;double&gt;
     * * [default=0, alias: reg_alpha]
     * * L1 regularization term on weights, increase this value will make model more conservative.</pre>
     * *
     * * <pre> -reg_alpha &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -tree_method &lt;string&gt;
     * * [default='auto'], The tree construction algorithm used in XGBoost; Choices: {'auto', 'exact', 'approx'} </pre>
     * *
     * * <pre> -sketch_eps &lt;double&gt;
     * * [default=0.03], range: (0, 1)
     * * Only used for approximate greedy algorithm. This roughly translated into O(1 / sketch_eps) number of bins. Compared to directly select number of bins, this comes with theoretical guarantee with sketch accuracy. Usually user does not have to tune this. but consider setting to a lower number for more accurate enumeration.</pre>
     * *
     * * <pre> -scale_pos_weight &lt;double&gt;
     * * [default=1]
     * * Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative cases) / sum(positive cases)</pre>
     * *
     * * <pre> -updater &lt;string&gt;
     * * [default='grow_colmaker,prune']
     * * A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters.</pre>
     * *
     * * <pre> -refresh_leaf &lt;integer&gt;
     * * [default=1]
     * * This is a parameter of the 'refresh' updater plugin. When this flag is true, tree leafs as well as tree nodes' stats are updated. When it is false, only node stats are updated.</pre>
     * *
     * * <pre> -process_type &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -sample_type &lt;string&gt;
     * * [default="uniform"]
     * * type of sampling algorithm:
     * * -"uniform": dropped trees are selected uniformly.
     * * -"weighted": dropped trees are selected in proportion to weight.</pre>
     * *
     * * <pre> -normalize_type &lt;string&gt;
     * * [default="tree"]
     * * type of normalization algorithm:
     * * -"tree": new trees have the same weight of each of dropped trees.
     * * -"forest": new trees have the same weight of sum of dropped trees (forest).</pre>
     * *
     * * <pre> -rate_drop &lt;double&gt;
     * * [default=0.0], range: [0.0, 1.0]
     * * dropout rate (a fraction of previous trees to drop during the dropout)</pre>
     * *
     * * <pre> -one_drop &lt;integer&gt;
     * * [default=0]
     * * when this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).</pre>
     * *
     * * <pre> -skip_drop &lt;double&gt;
     * * [default=0.0], range: [0.0, 1.0]
     * * Probability of skipping the dropout procedure during a boosting iteration.</pre>
     * *
     * * <pre> -lambda_bias &lt;double&gt;
     * * [default=0, alias: reg_lambda_bias]
     * * L2 regularization term on bias (no L1 reg on bias because it is not important)</pre>
     * *
     * * <pre> -reg_lambda_bias &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -objective &lt;string&gt;
     * * [ default=reg:linear ]
     * * "reg:linear" --linear regression
     * * "reg:logistic" --logistic regression
     * * "binary:logistic" --logistic regression for binary classification, output probability
     * * "binary:logitraw" --logistic regression for binary classification, output score before logistic transformation
     * * "count:poisson" --poisson regression for count data, output mean of poisson distribution [max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)]
     * * "multi:softmax" --set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
     * * "multi:softprob" --same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
     * * "rank:pairwise" --set XGBoost to do ranking task by minimizing the pairwise loss
     * * "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed
     * * "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.</pre>
     * *
     * * <pre> -num_class &lt;integer&gt;
     * * number of classes</pre>
     * *
     * * <pre> -base_score &lt;double&gt;
     * * [ default=0.5 ]
     * * -the initial prediction score of all instances, global bias
     * * -for sufficient number of iterations, changing this value will not have too much effect.</pre>
     * *
     * * <pre> -eval_metric &lt;string&gt;
     * * [ default according to objective ]
     * * evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )</pre>
     * *
     * * <pre> -seed &lt;integer&gt;
     * * [ default=0 ]
     * * random number seed.</pre>
     * *
     * * <pre> -tweedie_variance_power &lt;double&gt;
     * * [default=1.5], range: (1,2)
     * * parameter that controls the variance of the Tweedie distribution
     * * -set closer to 2 to shift towards a gamma distribution
     * * -set closer to 1 to shift towards a Poisson distribution.</pre>
     * *
     * * <pre> -output-debug-info
     * *  If set, classifier is run in debug mode and
     * *  may output additional info to the console</pre>
     * *
     * * <pre> -do-not-check-capabilities
     * *  If set, classifier capabilities are not checked before classifier is built
     * *  (use with caution).</pre>
     * *
     * * <pre> -num-decimal-places
     * *  The number of decimal places for the output of numbers in the model (default 2).</pre>
     * *
     * * <pre> -batch-size
     * *  The desired batch size for batch prediction  (default 100).</pre>
     * *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        super.setOptions(options);

        Integer num_round = getIntOptionValue("num_round", options);
        if (num_round != null) {
            this.numRound = num_round;
        }

        forceProbabilityDistribution = Utils.getFlag("force-probability-distribution", options);

        xgBoostParamsOptions.forEach(o -> checkOption(o, options));

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        Vector<String> options = new Vector<String>();

        if (forceProbabilityDistribution) {
            options.add("-force-probability-distribution");
        }

        options.add("-num_round");
        options.add(String.valueOf(this.numRound));


        params.forEach((name, val) -> {
            options.add("-" + name);
            options.add(String.valueOf(val));
        });

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    }


    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();   // returns the object from weka.classifiers.Classifier

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Returns a description of the classifier.
     *
     * @return a description of the classifier as a string.
     */
    /*@Override
    public String toString() {
        StringBuffer sb = new StringBuffer();
        sb.append("XGBoost Classifier: ");
        if (booster == null) {
            sb.append("No model built yet.");
        }else{

            String[] modelDump = new String[0];
            try {
                modelDump = booster.getModelDump(null, true);
            } catch (XGBoostError xgBoostError) {
                xgBoostError.printStackTrace();
                sb.append("Error dumping model: ").append(xgBoostError.getMessage());
            }
            sb.append(Arrays.asList(modelDump));
        }

        return sb.toString();
    }*/

    void checkOption(OptionWithType o, String[] options) {
        if (OptionWithType.ArgType.STRING.equals(o.type())) {
            checkStringOption(o.name(), options);
        } else if (OptionWithType.ArgType.INTEGER.equals(o.type())) {
            checkIntOption(o.name(), options);
        } else if (OptionWithType.ArgType.DOUBLE.equals(o.type())) {
            checkDoubleOption(o.name(), options);
        }
    }

    void checkStringOption(String name, String[] options) {
        String paramStr = getOptionValue(name, options);
        if (paramStr != null) {
            params.put(name, paramStr);
        }
    }

    void checkDoubleOption(String name, String[] options) {
        String paramStr = getOptionValue(name, options);
        if (paramStr != null) {
            params.put(name, Double.parseDouble(paramStr));
        }
    }

    void checkIntOption(String name, String[] options) {
        Integer param = getIntOptionValue(name, options);
        if (param != null) {
            params.put(name, param);
        }
    }

    Integer getIntOptionValue(String name, String[] options) {
        String paramStr = getOptionValue(name, options);
        if (paramStr == null) {
            return null;
        }
        return Integer.parseInt(paramStr);
    }

    String getOptionValue(String name, String[] options) {
        try {
            String option = Utils.getOption(name, options).trim();
            return option.isEmpty() ? null : option;
        } catch (Exception e) {
            return null;
        }
    }

    static void addStringOption(String name) {
        addStringOption(name, "");
    }

    static void addStringOption(String name, String description) {
        xgBoostParamsOptions.add(createOption(name, description, OptionWithType.ArgType.STRING));
    }

    static void addIntOption(String name) {
        addIntOption(name, "");
    }

    static void addIntOption(String name, String description) {
        xgBoostParamsOptions.add(createOption(name, description, OptionWithType.ArgType.INTEGER));
    }

    static void addDoubleOption(String name) {
        addDoubleOption(name, "");
    }

    static void addDoubleOption(String name, String description) {
        xgBoostParamsOptions.add(createOption(name, description, OptionWithType.ArgType.DOUBLE));
    }

    static OptionWithType createOption(String name, String description, OptionWithType.ArgType argType) {
        String synopsis = "-" + name + " <" + argType.name().toLowerCase() + ">";
        return new OptionWithType(description == null ? name : description, name, 1, synopsis, argType);
    }


    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        XGBoost classifier = new XGBoost();
        runClassifier(classifier, argv);
        System.out.println(classifier);
    }


    @Override
    public TechnicalInformation getTechnicalInformation() {
        return new TechnicalInformation(TechnicalInformation.Type.MISC);
    }
}