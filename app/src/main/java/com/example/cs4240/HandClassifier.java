package com.example.cs4240;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomLocalModel;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelInterpreterOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class HandClassifier {

    private static final int inputSize = 256;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private int outputSize;
    //private Interpreter interpreter;
    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private Activity parentActivity;
    private float[][] results;

    public HandClassifier(Activity activity, String model, int outputSize) throws IOException {
        this.outputSize = outputSize;
        this.parentActivity = activity;
        this.results = null;

        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath(model)
                .build();
        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);

            inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 256, 256, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 42})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }
    }

    public void predict(Bitmap image) throws FirebaseMLException {
        long startTime = System.nanoTime();

        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                .add(convertBitmapToByteBuffer(image))  // add() as many input arrays as your model requires
                .build();
        interpreter.run(inputs, inputOutputOptions)
                .addOnSuccessListener(
                        new OnSuccessListener<FirebaseModelOutputs>() {
                            @Override
                            public void onSuccess(FirebaseModelOutputs result) {
                                Log.d("test", "success");
                                results = result.getOutput(0);
                                ((CameraActivity)parentActivity).labelImage();
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                e.printStackTrace();
                            }
                        });

        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
        Log.d("test", "Time taken to predict: " + String.valueOf(duration/1000000) + "ms");
    }

    public void label(Bitmap image) {
        float xScale = image.getWidth() / (float)inputSize;
        float yScale = image.getHeight() / (float)inputSize;

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.FILL);

        Canvas canvas = new Canvas(image);
        float minX = 9999;
        float minY = 9999;
        float maxX = 0;
        float maxY = 0;
        for (int i = 0; i < outputSize; i+=2) {
            // Get the min x,y and max x,y to create roi for hand
            float xCoord = results[0][i] * xScale;
            float yCoord = results[0][i+1] * yScale;
            if (xCoord < minX)
                minX = xCoord;
            if (xCoord > maxX)
                maxX = xCoord;

            if (yCoord < minY)
                minY = yCoord;
            if (yCoord > maxY)
                maxY = yCoord;

            // Hand keypoints
            canvas.drawCircle(results[0][i] * xScale, results[0][i+1] * yScale, 10.0f, paint);
        }

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5.0f);
        canvas.drawRect(minX -50, minY-50, maxX+50, maxY+50, paint);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    public float getSigM(float x) {
        return (float) ((float) 1 / (1 + Math.exp(-x)));
    }
}
