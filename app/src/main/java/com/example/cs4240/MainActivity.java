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
    private static final String sign_model_file = "converted_model.tflite";

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

            classifier = new HandClassifier(this, model_file);
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
        //Intent i = new Intent(this, CameraActivity.class);
        //startActivity(i);

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap bmp = BitmapFactory.decodeResource(this.getResources(),  test_img, options);

        Log.d("test", "Before cvt Bmp Size:"+ bmp.getWidth() + "x" + bmp.getHeight());
        Mat src = new Mat();
        Utils.bitmapToMat(bmp, src);
        Mat dest = new Mat();

        Imgproc.cvtColor(src, dest, Imgproc.COLOR_BGR2GRAY);

        Bitmap gbmp = Bitmap.createBitmap(dest.cols(), dest.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dest, gbmp);
        Log.d("test", "After cvt Bmp Size:"+ gbmp.getWidth() + "x" + gbmp.getHeight());

        ImageView imView = (ImageView)findViewById(R.id.im_view);
        imView.setImageBitmap(gbmp);

        try {
            signClassifier.predict(gbmp);
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        bmpOriginal.recycle();
        return bmpGrayscale;
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
