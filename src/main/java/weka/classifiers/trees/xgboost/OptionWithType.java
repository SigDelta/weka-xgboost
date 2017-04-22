package weka.classifiers.trees.xgboost;

import weka.core.Option;

import java.io.Serializable;


public class OptionWithType extends Option implements Serializable {

    private static final long serialVersionUID = -251269918815312469L;

    public enum ArgType{
        STRING, DOUBLE, INTEGER
    }

    private final ArgType type;

    /**
     * Creates new option with the given parameters.
     *
     * @param description  the option's description
     * @param name         the option's name
     * @param numArguments the number of arguments
     * @param synopsis     the option's synopsis
     * @param type         the option's arg type
     */
    public OptionWithType(String description, String name, int numArguments, String synopsis, ArgType type) {
        super(description, name, numArguments, synopsis);
        this.type = type;
    }

    public ArgType type(){
        return type;
    }
}
