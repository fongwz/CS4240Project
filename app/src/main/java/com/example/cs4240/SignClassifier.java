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
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 39})
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
                                getDetections(result); //post processing
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

    private void getDetections(FirebaseModelOutputs firebaseResults) {
        results = firebaseResults.getOutput(0);

        Log.d("test", Arrays.deepToString(results));
        Log.d("test", "idx: " + argMax(results, 0) + " : value " + textArr[argMax(results, 0)]);
    }

    private void initTextArr() {
        textArr = new String[]{"a", "b", "c", "d", "delete", "e", "f", "g", "h", "i",
                                "j", "k", "l", "m", "n", "nothing", "o", "p", "q", "r",
                                "s", "space", "t", "u", "v", "w", "x", "y", "z", "0",
                                "1", "2", "3", "4", "5", "6", "7", "8", "9"};
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
