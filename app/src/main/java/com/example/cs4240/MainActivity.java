package com.example.cs4240;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private static final int test_img = R.drawable.i;
    private static final String model_file = "palm.tflite";
    private static final String sign_model_file = "default.tflite";

    private HandClassifier classifier;
    private SignClassifier signClassifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            Bitmap bmp = BitmapFactory.decodeResource(this.getResources(), test_img);
            Bitmap newBmp = bmp.copy(android.graphics.Bitmap.Config.ARGB_8888, true);

            signClassifier = new SignClassifier(this, sign_model_file);
            //signClassifier = new SignClassifier(this, sign_model_file);
            //classifier.predict(toGrayscale(bmp));

            //Bitmap grayscaleBmp = toGrayscale(bmp);
            signClassifier.predict(bmp);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setImage() {
        Bitmap bmp = BitmapFactory.decodeResource(this.getResources(), test_img);
        Bitmap newBmp = bmp.copy(android.graphics.Bitmap.Config.ARGB_8888, true);
        //this.classifier.label(newBmp);
        //ImageView imView = (ImageView)findViewById(R.id.im_view);
        //imView.setImageBitmap(newBmp);

    }

    public void onButtonClick(View view) {
        Intent i = new Intent(this, CameraActivity.class);
        startActivity(i);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("test", "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };
}
