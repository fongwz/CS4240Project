package com.example.cs4240;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;

public class SignClassifier {

    private static final int inputSize = 70;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 1;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private Activity parentActivity;
    private float[][] results;
    private float[][] res_clf;
    private boolean[] clf_mask;
    private float[][] res_reg;
    private ByteBuffer byteBuffer;

    private CSVReader csvReader;
    private ArrayList<float[]> anchors;
    private ArrayList<float[]> anchorCandidates;

    private float dx;
    private float dy;
    private float w;
    private float h;
    private float cx;
    private float cy;

    private String displayText = "";
    private String[] textArr;

    public SignClassifier(Activity activity, String model) throws IOException {
        this.parentActivity = activity;
        this.results = null;
        this.byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        this.clf_mask = new boolean[2944];
        this.csvReader = new CSVReader(new BufferedReader(new InputStreamReader(activity.getAssets().open("anchors.csv"))));
        this.anchors = new ArrayList<>();
        this.anchorCandidates = new ArrayList<>();
        loadAnchors();
        initTextArr();

        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(model)
                .build();
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);

            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, inputSize, inputSize, PIXEL_SIZE})
//                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 2944, 18})
//                            .setOutputFormat(1, FirebaseModelDataType.FLOAT32, new int[]{1, 2944, 1})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 39})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    public void predict(Bitmap image) throws FirebaseMLException {
        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                .add(convertBitmapToByteBuffer(image))  // add() as many input arrays as your model requires
                .build();
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                Log.d("test", "sign reading success");
                                getDetections(result); //post processing
                                ((CameraActivity)parentActivity).setTextImage();
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                e.printStackTrace();
                            }
                        });
    }

    public void label(Bitmap image) {
        /*
        Labelling method for signs
         */
        Canvas canvas = new Canvas(image);

        Paint paint = new Paint();

        paint.setColor(Color.WHITE);
        paint.setTextSize(20);
        Log.d("test", "displaying " + displayText);
        canvas.drawText(displayText, 10, 25, paint);
    }

    private void loadAnchors() {
        String[] nextLine;
        try {
            while ((nextLine = csvReader.readNext()) != null) {
                anchors.add(new float[]{
                        Float.valueOf(nextLine[0]),
                        Float.valueOf(nextLine[1]),
                        Float.valueOf(nextLine[2]),
                        Float.valueOf(nextLine[3])
                });
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        byteBuffer.clear();
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
//                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
//                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((val-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    private void getDetections(FirebaseModelOutputs firebaseResults) {
        res_reg = firebaseResults.getOutput(0);
//        res_clf = firebaseResults.getOutput(1);

        Log.d("test", Arrays.deepToString(res_reg));
        anchorCandidates.clear();

        for (int i = 0; i < res_reg[0].length; i++) {
            if (res_reg[0][i] == 1.0f) {
                Log.d("test", "idx: " + i + " : value " + textArr[i]);
                this.displayText = textArr[i];
//                if (displayText.isEmpty()){
//                    displayText.concat(textArr[i]);
//                } else {
//                    displayText
//                }
            }
        }
//        Log.d("test", "post processing sign model output");

//        createDetectionMask();
//        createFilteredDetections();
//        int maxIdx = argMax(detectionCandidates, 3);
//
//        if (detectionCandidates.size() <= 0) {
//            return;
//        }
    }

    private void initTextArr() {
        textArr = new String[]{"a", "b", "c", "d", "delete", "e", "f", "g", "h", "i",
                                "j", "k", "l", "m", "n", "nothing", "o", "p", "q", "r",
                                "s", "space", "t", "u", "v", "w", "x", "y", "z", "0",
                                "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    }
//
//    public void createDetectionMask() {
//        for (int i = 0 ; i < res_clf[0].length; i++) {
//            clf_mask[i] = getSigM(res_clf[0][i][0]) > 0.7;
//        }
//    }
//
//    public float getSigM(float x) {
//        return (float) ((float) 1 / (1 + Math.exp(-x)));
//    }
//
//    public void createFilteredDetections() {
//        for (int i = 0; i < res_reg[0].length; i++) {
//            if (clf_mask[i]) {
//                detectionCandidates.add(res_reg[0][i]);
//                anchorCandidates.add(anchors.get(i));
//            }
//        }
//    }
//
//    /*
//    Argmax for a specific index in a 2d array
//     */
//    public int argMax(ArrayList<float[]> array, int arrayIndexSelector) {
//        int idx = 0;
//        float currMax = -999;
//        for (int i = 0 ; i < array.size(); i++) {
//            if (array.get(i)[arrayIndexSelector] > currMax) {
//                idx = i;
//                currMax = array.get(i)[arrayIndexSelector];
//            }
//        }
//
//        return idx;
//    }
}
