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
import java.util.HashMap;
import java.util.Map;

public class HandClassifier {

    private static final int inputSize = 256;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private int outputSize;
    private Interpreter interpreter;
    //private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private Activity parentActivity;
    //private float[][] results;
    private float[][] results2;
    private ByteBuffer byteBuffer;

    private long startTime;
    private Map<Integer, Object> results = new HashMap<>();
    private Object[] inputs = new Object[1];

    public HandClassifier(Activity activity, String model, int outputSize) throws IOException {
        this.outputSize = outputSize;
        this.parentActivity = activity;
        //this.results = null;
        this.byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);

        /*
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
                            .setOutputFormat(1, FirebaseModelDataType.FLOAT32, new int[]{1, 1})
                            .build();
        } catch (FirebaseMLException e) {
            e.printStackTrace();
        }*/

        this.interpreter = new Interpreter(loadModelFile(activity, "hand_landmark.tflite"));
        float[][] res1 = new float[1][42];
        float[][] res2 = new float[1][1];
        this.results.put(0, res1);
        this.results.put(1, res2);
    }

    public void predict(Bitmap image) {
        startTime = System.nanoTime();
        /*
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
                                results2 = result.getOutput(1);
                                Log.d("test", "opop: " + String.valueOf(results2[0][0]));
                                long endTime = System.nanoTime();
                                long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds.
                                Log.d("test", "Time taken to predict: " + String.valueOf(duration/1000000) + "ms");
                                if (results2[0][0] >= 0.9)
                                    ((CameraActivity)parentActivity).setImage();
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {
                                e.printStackTrace();
                            }
                        });
        */

        inputs[0] = convertBitmapToByteBuffer(image);
        interpreter.runForMultipleInputsOutputs(inputs, results);
        Log.d("test", String.valueOf(((float[][])results.get(1))[0][0]));
    }

    public void label(Bitmap image) {

        if (((float[][])results.get(1))[0][0] < 0.3) {
            return;
        }

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
            float xCoord = ((float[][])results.get(0))[0][i] * xScale;
            float yCoord = ((float[][])results.get(0))[0][i+1] * yScale;
            if (xCoord < minX)
                minX = xCoord;
            if (xCoord > maxX)
                maxX = xCoord;

            if (yCoord < minY)
                minY = yCoord;
            if (yCoord > maxY)
                maxY = yCoord;

            // Hand keypoints
            canvas.drawCircle(((float[][])results.get(0))[0][i] * xScale, ((float[][])results.get(0))[0][i+1] * yScale, 10.0f, paint);
        }
        Log.d("test", String.valueOf(minX));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(15.0f);
        paint.setColor(Color.WHITE);
        canvas.drawRect(minX -50, minY-50, maxX+50, maxY+50, paint);
    }

    private MappedByteBuffer loadModelFile(Activity activity,String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        byteBuffer.clear();
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
