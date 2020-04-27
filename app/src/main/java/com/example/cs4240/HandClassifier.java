package com.example.cs4240;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
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
import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;

public class HandClassifier {

    private static final int inputSize = 256;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    private FirebaseModelInterpreter interpreter;
    private FirebaseModelInputOutputOptions inputOutputOptions;
    private Activity parentActivity;
    private float[][] results;
    private float[][][] res_clf;
    private boolean[] clf_mask;
    private float[][][] res_reg;
    private ArrayList<float[]> detectionCandidates;
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

    private int x;
    private int y;
    private int width;
    private int height;

    private int counter =0;


    private long startTime;

    public HandClassifier(Activity activity, String model) throws IOException {
        this.parentActivity = activity;
        this.results = null;
        this.byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        this.detectionCandidates = new ArrayList<>();
        this.clf_mask = new boolean[2944];
        this.csvReader = new CSVReader(new BufferedReader(new InputStreamReader(activity.getAssets().open("anchors.csv"))));
        this.anchors = new ArrayList<>();
        this.anchorCandidates = new ArrayList<>();
        loadAnchors();

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
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 2944, 18})
                            .setOutputFormat(1, FirebaseModelDataType.FLOAT32, new int[]{1, 2944, 1})
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
                                getDetections(result);
                                //((MainActivity)parentActivity).setImage();
                                calculateParameters(((CameraActivity)parentActivity).getBitmap());
                                ((CameraActivity)parentActivity).predictSign(x, y, width, height);
                                //results2 = result.getOutput(1);
                                //Log.d("test", "Time taken to process results: " + String.valueOf(duration/1000000) + "ms");
                                //if (results2[0][0] >= 0.9)
                                //    ((CameraActivity)parentActivity).setImage();
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

        /* Labelling method for hand_landmark
        if (results[0][0] < 0.3) {
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
        for (int i = 0; i < 42; i+=2) {
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
            canvas.drawCircle(xCoord, yCoord, 10.0f, paint);
        }
        Log.d("test", String.valueOf(minX));
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(15.0f);
        paint.setColor(Color.WHITE);
        canvas.drawRect(minX -50, minY-50, maxX+50, maxY+50, paint);
         */

        /*
        Labelling method for palm
         */
        if (detectionCandidates.size() <= 0) {
            return;
        }

        float xScale = image.getWidth() / (float)inputSize;
        float yScale = image.getHeight() / (float)inputSize;

        float newW = yScale * w;
        float newH = xScale * h;

        Paint paint = new Paint();
        Canvas canvas = new Canvas(image);

        paint.setColor(Color.BLACK);
        canvas.drawCircle(cx, cy, 10.0f, paint); // anchor or center of palm
        paint.setColor(Color.GREEN);

//        canvas.drawCircle(cx-dx, cy-dy, 10.0f, paint);
//        canvas.drawCircle(cx-dx, cy+dy, 10.0f, paint);
//        canvas.drawCircle(cx+dx, cy-dy, 10.0f, paint);
//        canvas.drawCircle(cx+dx, cy+dy, 10.0f, paint);
        canvas.drawCircle((cx-newW-dx), (cy-newH-dy), 10.0f, paint);

        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(5.0f);

        canvas.drawRect((cx-newW-dx), (cy-newH-dy), (cx+newW-dx), (cy+newH-dy), paint); //left, top, right, bottom

//        if (counter == 20) {
//            Bitmap bmp = ((CameraActivity)parentActivity).getBitmap();
//            Bitmap roi = Bitmap.createBitmap(bmp, x, y, width, height);
//            canvas.drawBitmap(roi, 0, 0, );
//            counter = 0;
//        }
    }

    private void calculateParameters(Bitmap image) {
        if (detectionCandidates.size() <= 0) {
            return;
        }
        float xScale = image.getWidth() / (float)inputSize;
        float yScale = image.getHeight() / (float)inputSize;

        this.width = (int)(xScale * w);
        this.height = (int)(yScale * h);
        this.x = (int)(cx-width-dx);
        this.y = (int)(cy-height-dy);
    }

    private MappedByteBuffer loadModelFile(Activity activity,String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
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

    public void getDetections(FirebaseModelOutputs firebaseResults) {
        res_reg = firebaseResults.getOutput(0);
        res_clf = firebaseResults.getOutput(1);

        detectionCandidates.clear();
        anchorCandidates.clear();

        createDetectionMask();
        createFilteredDetections();
        int maxIdx = argMax(detectionCandidates, 3);

        if (detectionCandidates.size() <= 0) {
            return;
        }

//        Log.d("test", "arr size : " + detectionCandidates.size());
//        Log.d("test", Arrays.deepToString(detectionCandidates.toArray()));
        dx = detectionCandidates.get(maxIdx)[0];
        dy = detectionCandidates.get(maxIdx)[1];
        w = detectionCandidates.get(maxIdx)[2];
        h = detectionCandidates.get(maxIdx)[3];

        cx = anchorCandidates.get(maxIdx)[0] * 720; //256
        cy = anchorCandidates.get(maxIdx)[1] * 720; //256

//        Log.d("detection result", dx + " : " + dy + " : " + w + " : " + h + " : " + cx + " : " + cy);
    }

    public void createDetectionMask() {
        for (int i = 0 ; i < res_clf[0].length; i++) {
            clf_mask[i] = getSigM(res_clf[0][i][0]) > 0.7;
        }
    }

    public void createFilteredDetections() {
        for (int i = 0; i < res_reg[0].length; i++) {
            if (clf_mask[i]) {
                detectionCandidates.add(res_reg[0][i]);
                anchorCandidates.add(anchors.get(i));
            }
        }
    }

    public float getSigM(float x) {
        return (float) ((float) 1 / (1 + Math.exp(-x)));
    }


    /*
    Argmax for a specific index in a 2d array
     */
    public int argMax(ArrayList<float[]> array, int arrayIndexSelector) {
        int idx = 0;
        float currMax = -999;
        for (int i = 0 ; i < array.size(); i++) {
            if (array.get(i)[arrayIndexSelector] > currMax) {
                idx = i;
                currMax = array.get(i)[arrayIndexSelector];
            }
        }

        return idx;
    }
}
