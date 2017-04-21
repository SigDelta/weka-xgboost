package weka.classifiers.trees.xgboost;


import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

public class DMatrixLoader {

    public static DMatrix instancesToDMatrix(Instances instances) throws XGBoostError {
        long[] rowHeaders = new long[instances.size()+1];
        rowHeaders[0]=0;
        List<Float> dataList = new ArrayList<>();
        List<Integer> colList = new ArrayList<>();
        float[] labels = new float[instances.size()];

        for(int i=0; i<instances.size(); i++) {
            Instance instance = instances.get(i);
            rowHeaders[i] = dataList.size();
            processInstance(instance, dataList, colList);
            labels[i] = (float) instance.classValue();
        }
        rowHeaders[rowHeaders.length - 1] = dataList.size();
        int colNum = instances.numAttributes()-1;
        DMatrix dMatrix = createDMatrix(rowHeaders, dataList, colList, colNum);

        dMatrix.setLabel(labels);
        return dMatrix;

    }

    public static DMatrix instancesToDenseDMatrix(Instances instances) throws XGBoostError {
        int colNum = instances.numAttributes()-1;
        int rowNum = instances.size();

        float[] data = new float[colNum*rowNum];
        float[] labels = new float[instances.size()];
        Attribute classAttribute = instances.classAttribute();
        int classAttrIndex = classAttribute.index();

        for(int i=0, dataIndex = 0; i<instances.size(); i++) {
            Instance instance = instances.get(i);

            labels[i] = (float) instance.classValue();
            Enumeration<Attribute> attributeEnumeration = instance.enumerateAttributes();
            while (attributeEnumeration.hasMoreElements()) {
                Attribute attribute = attributeEnumeration.nextElement();
                int attrIndex = attribute.index();
                if(attrIndex == classAttrIndex){
                    continue;
                }
                data[dataIndex]= (float) instance.value(attribute);
                dataIndex++;
            }
        }


        DMatrix dMatrix = new DMatrix(data, rowNum, colNum);

        dMatrix.setLabel(labels);
        return dMatrix;

    }

    public static DMatrix instanceToDenseDMatrix(Instance instance) throws XGBoostError {
        Attribute classAttribute = instance.classAttribute();
        int classAttrIndex = classAttribute.index();

        int colNum = instance.numAttributes()-1;
        int rowNum = 1;

        float[] data = new float[colNum*rowNum];

        Enumeration<Attribute> attributeEnumeration = instance.enumerateAttributes();
        int dataIndex = 0;
        while (attributeEnumeration.hasMoreElements()) {
            Attribute attribute = attributeEnumeration.nextElement();
            int attrIndex = attribute.index();
            if(attrIndex == classAttrIndex){
                continue;
            }
            data[dataIndex]= (float) instance.value(attribute);
            dataIndex++;
        }

        return new DMatrix(data, rowNum, colNum);
    }

    public static DMatrix instanceToDMatrix(Instance instance) throws XGBoostError {

        List<Float> dataList = new ArrayList<>();
        List<Integer> colList = new ArrayList<>();

        processInstance(instance, dataList, colList);
        long[] rowHeaders = new long[]{0, dataList.size()};

        int colNum = instance.numAttributes()-1;
        return createDMatrix(rowHeaders, dataList, colList, colNum);
    }

    protected static DMatrix createDMatrix(long[] rowHeaders, List<Float> dataList, List<Integer> colList, int colNum) throws XGBoostError {
        float[] data = new float[dataList.size()];
        int[] colIndices = new int[dataList.size()];
        colIndices[0] = 0;
        for(int i=0; i<dataList.size(); i++) {
            data[i] = dataList.get(i);
            colIndices[i] = colList.get(i);
        }

        return new DMatrix(rowHeaders, colIndices, data, DMatrix.SparseType.CSR, colNum);
    }

    protected static void processInstance(Instance instance, List<Float> dataList, List<Integer> colList ){
        Attribute classAttribute = instance.classAttribute();
        int classAttrIndex = classAttribute.index();
        Enumeration<Attribute> attributeEnumeration = instance.enumerateAttributes();
        while (attributeEnumeration.hasMoreElements()) {
            Attribute attribute = attributeEnumeration.nextElement();
//            System.out.print(attribute.name()+", ");
            int attrIndex = attribute.index();

            if(attrIndex == classAttrIndex){
                continue;
            }
            double value = instance.value(attribute);

            if (value == 0) {
                continue;
            }

            dataList.add((float) value);

            if (attrIndex < classAttrIndex) {
                colList.add(attrIndex);
            }else{
                colList.add(attrIndex+1);
            }

        }
//        System.out.println();
    }
}
