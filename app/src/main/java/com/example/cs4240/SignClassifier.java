package com.example.cs4240;

import android.app.Activity;
import android.graphics.Bitmap;
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

import org.apache.commons.lang3.ObjectUtils;

import java.io.IOException;
import java.util.Arrays;

public class SignClassifier {

    private static final int inputSize = 70;

    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private Activity parentActivity;
    private float[][] results;
    private float[][][][] input;

    private String displayText = "";
    private String[] textArr;

    private int[] detectResults = new int[7];
    private int counter = 0;

    public SignClassifier(Activity activity, String model) throws IOException {
        this.parentActivity = activity;
        this.results = null;
        initTextArr();
        input = new float[1][inputSize][inputSize][1];

        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(model)
                .build();
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);

            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, inputSize, inputSize, 1})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 26})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    public void predict(Bitmap image) throws FirebaseMLException {
        Bitmap scaledImg = Bitmap.createScaledBitmap(image, inputSize, inputSize, false);

        for (int x = 0; x < inputSize; x++) {
            for (int y = 0; y < inputSize; y++) {
                int pixel = scaledImg.getPixel(y, x);
                // Color to grey: R+G+B / num channels
                // Normalize img to [0.0 - 1.0]: grey / 255.0f
                input[0][x][y][0] = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / (3.0f * 255.0f);
            }
        }

        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                .add(input)  // add() as many input arrays as your model requires
                .build();
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                //try {
                                    int toDetect = getDetections(result); //post processing
                                    getDetectedResult(toDetect);
                                //} catch (NullPointerException e){
                                 //   System.out.println("Null pointer exception caught");
                                //}
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                Log.d("test", "cannot read image");
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
        paint.setTextSize(100);
        Log.d("test", "displaying " + displayText);
        canvas.drawText(displayText, 100, 100, paint);
    }

    private void getDetectedResult(int result){
        detectResults[counter] = result;  //store results in detectResults[] for comparison later on
        counter++;

        //Array is filled
        if (counter == 7){
            Arrays.sort(detectResults);

            // find the max frequency using linear traversal
            int max_count = 1, res = detectResults[0];
            int curr_count = 1;

            for (int i = 1; i < 7; i++)
            {
                if (detectResults[i] == detectResults[i - 1])
                    curr_count++;
                else
                {
                    if (curr_count > max_count)
                    {
                        max_count = curr_count;
                        res = detectResults[i - 1];
                    }
                    curr_count = 1;
                }
            }

            // If last element is most frequent
            if (curr_count > max_count)
            {
                max_count = curr_count;
                res = detectResults[6];
            }
            if (max_count >= 4) { //if appear more than 4 times, character appears here
                Log.d("test", "Detected letter is: " + textArr[res]);
            }
            counter = 0; //reset counter
        }
    }

    private int getDetections(FirebaseModelOutputs firebaseResults) {
        results = firebaseResults.getOutput(0);

        //Log.d("test", Arrays.deepToString(results));
        Log.d("test", "idx: " + argMax(results, 0) + " : value " + textArr[argMax(results, 0)]);
        ((Camera2Activity) parentActivity).togglePredicting();
        return argMax(results, 0);
    }

    private void initTextArr() {
        textArr = new String[]{"a", "b", "c", "d", "e", "f", "g", "h", "i",
                                "j", "k", "l", "m", "n", "o", "p", "q", "r",
                                "s", "t", "u", "v", "w", "x", "y", "z"};
    }

    public int argMax(float[][] array, int arrayIndexSelector) {
        int idx = 0;
        float currMax = -999;
        for (int i = 0 ; i < array[arrayIndexSelector].length; i++) {
            if (array[arrayIndexSelector][i] > currMax) {
                idx = i;
                currMax = array[arrayIndexSelector][i];
            }
        }
        return idx;
    }
}
