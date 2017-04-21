package weka.classifiers.trees.xgboost;

import ml.dmlc.xgboost4j.java.Booster;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import ml.dmlc.xgboost4j.java.DMatrix;
import weka.core.Capabilities.Capability;

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
 * * <pre> -booster &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -silent &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -nthread &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -num_pbuffer &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -num_feature &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -eta &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -gamma &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -max_depth &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -min_child_weight &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -max_delta_step &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -subsample &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -colsample_bytree &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -colsample_bylevel &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -lambda &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -alpha &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -tree_method &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -sketch_eps &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -scale_pos_weight &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -updater &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -refresh_leaf &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -process_type &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -sample_type &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -normalize_type &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -rate_drop &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -one_drop &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -skip_drop &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -lambda_bias &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -objective &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -num_class &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -base_score &lt;double&gt;
 * * </pre>
 * *
 * * <pre> -eval_metric &lt;string&gt;
 * * </pre>
 * *
 * * <pre> -seed &lt;integer&gt;
 * * </pre>
 * *
 * * <pre> -tweedie_variance_power &lt;double&gt;
 * * </pre>
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
public class XGBoost extends AbstractClassifier implements TechnicalInformationHandler {

    private Booster booster;

    Map<String, Object> params = new HashMap<>();

    static List<OptionWithType> xgBoostParamsOptions = new ArrayList<>();

    static Set<String> probabilityObjective = new HashSet<>(Arrays.asList("binary:logistic", "multi:softprob"));

    private boolean forceProbabilityDistribution = false;

    static {
        // General Parameters
        addStringOption("booster");
        addIntOption("silent");
        addIntOption("nthread");
        addIntOption("num_pbuffer");
        addIntOption("num_feature");

        // Parameters for Tree Booster
        addDoubleOption("eta");
        addDoubleOption("gamma");
        addIntOption("max_depth");
        addDoubleOption("min_child_weight");
        addDoubleOption("max_delta_step");
        addDoubleOption("subsample");
        addDoubleOption("colsample_bytree");
        addDoubleOption("colsample_bylevel");
        addDoubleOption("lambda");
        addDoubleOption("alpha");
        addStringOption("tree_method");
        addDoubleOption("sketch_eps");
        addDoubleOption("scale_pos_weight");
        addStringOption("updater");
        addIntOption("refresh_leaf");
        addStringOption("process_type");

        // Additional parameters for Dart Booster
        addStringOption("sample_type");
        addStringOption("normalize_type");
        addDoubleOption("rate_drop");
        addIntOption("one_drop");
        addDoubleOption("skip_drop");

        //Additional parameters for Linear Booster
//        addDoubleOption("lambda");
//        addDoubleOption("alpha");
        addDoubleOption("lambda_bias");

        //Learning Task Parameters
        addStringOption("objective");
        addIntOption("num_class");
        addDoubleOption("base_score");
        addStringOption("eval_metric");
        addIntOption("seed");

        //Parameters for Tweedie Regression
        addDoubleOption("tweedie_variance_power");

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

        xgBoostParamsOptions.forEach(newVector::add);

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
     * * <pre> -booster &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -silent &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -nthread &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -num_pbuffer &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -num_feature &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -eta &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -gamma &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -max_depth &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -min_child_weight &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -max_delta_step &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -subsample &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -colsample_bytree &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -colsample_bylevel &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -lambda &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -alpha &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -tree_method &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -sketch_eps &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -scale_pos_weight &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -updater &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -refresh_leaf &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -process_type &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -sample_type &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -normalize_type &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -rate_drop &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -one_drop &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -skip_drop &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -lambda_bias &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -objective &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -num_class &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -base_score &lt;double&gt;
     * * </pre>
     * *
     * * <pre> -eval_metric &lt;string&gt;
     * * </pre>
     * *
     * * <pre> -seed &lt;integer&gt;
     * * </pre>
     * *
     * * <pre> -tweedie_variance_power &lt;double&gt;
     * * </pre>
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

        params.forEach((name, val) -> {
            options.add("-" + name + " " + val);
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