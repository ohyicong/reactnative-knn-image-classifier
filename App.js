import React, { useState, useEffect }  from 'react';
import { StyleSheet, View,Image,TouchableOpacity,Dimensions,ActivityIndicator } from 'react-native';
import { Text,Button, Input,Card,Overlay  } from 'react-native-elements';
import { Camera } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as jpeg from 'jpeg-js'
export default function App() {
  const statusList=["Loading Model...","Classifying Image...","Predicting Image..."]
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [isTfReady,setIsTfReady] = useState(false)
  const [mobilenetModel,setMobilenetModel] = useState(null)
  const [knnClassifierModel,setKnnClassifierModel] = useState(null)
  const [prediction,setPrediction] = useState({
    "label":"No Results",
    "confidence":{}
  })
  const [status,setStatus]=useState(statusList[0])
  const [isLoading,setIsLoading]=useState(true)
  const [countExamples,setCountExamples] = useState(0)
  const [countClassExamples,setCountClassExamples] = useState({
    "Class A":0,
    "Class B":0,
    "Class C":0
  })
  const classList=[
    {
      id:0,
      name:"Class A"
    },
    {
      id:1,
      name:"Class B"
    },
    {
      id:2,
      name:"Class C"
    },
  ]
  //load tensorflow 
  useEffect(() => {
    async function startup (){
      if(!isTfReady){
        console.log("[+] Loading TF Model")
        setStatus(statusList[0])
        setIsLoading(true)
        let { status } = await Camera.requestPermissionsAsync();
        setHasPermission(status === 'granted');
        await tf.ready()
        setIsTfReady(true)
        setMobilenetModel(await mobilenet.load())
        setKnnClassifierModel(await knnClassifier.create())
        setIsLoading(false)
        console.log("[+] TF Model Loaded")
      }
    }
    startup()
  },[isTfReady]);

  //1. collect and label images from camera
  const collectData = async(className)=>{
    console.log(`[+] Class ${className} selected`)
    setStatus(statusList[1])
    setIsLoading(true)
    try{
      if(this.camera){
        let photo = await this.camera.takePictureAsync({
          skipProcessing: true,
        });
        //2. resize images into width:224 height:224
        image = await resizeImage(photo.uri, 224 , 224);
        let imageTensor = base64ImageToTensor(image.base64);
        //3. get embeddings from mobilenet
        let embeddings = await mobilenetModel.infer(imageTensor, true);
        //4. train knn classifier
        knnClassifierModel.addExample(embeddings,className)
        let tempCountExamples = countExamples + 1
        let tempCountClassExamples = countClassExamples
        tempCountClassExamples[`${className}`] = tempCountClassExamples[`${className}`] +1 
        setCountExamples(tempCountExamples)
        setCountClassExamples(tempCountClassExamples)
  
        console.log("[+] Class Added")
  
      }
    }catch{
      console.log("[-] No Camera")
    }
    
    setIsLoading(false)
  } 
  //5. predict new images
  const getPredictions = async() =>{
    console.log("[+] Analysing Photo")
    setStatus(statusList[2])
    setIsLoading(true)
    try{
      if(this.camera){
        let photo = await this.camera.takePictureAsync({
          skipProcessing: true,
        });
        //resize images into width:224 height:224
        image = await resizeImage(photo.uri, 224 , 224);
        let imageTensor = base64ImageToTensor(image.base64);
        //get embeddings from mobilenet
        let embeddings = await mobilenetModel.infer(imageTensor,true)
        //predict with knn classifier
        let prediction = await knnClassifierModel.predictClass(embeddings);
        console.log(JSON.stringify(prediction))
        setPrediction(prediction)
        
      }
    }
    catch{
      console.log("[-] No Camera")
    }
    setIsLoading(false)
    console.log("[+] Photo Analysed")
  }
  function base64ImageToTensor(base64){
    //Function to convert jpeg image to tensors
    const rawImageData = tf.util.encodeString(base64, 'base64');
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3);
    let offset = 0; // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];
      offset += 4;
    }
    return tf.tensor3d(buffer, [height, width, 3]);
  }

  async function resizeImage(imageUrl, width, height){
    const actions = [{
      resize: {
        width,
        height
      },
    }];
    const saveOptions = {
      compress: 0.75,
      format: ImageManipulator.SaveFormat.JPEG,
      base64: true,
    };
    const res = await ImageManipulator.manipulateAsync(imageUrl, actions, saveOptions);
    return res;
  }

  return (
    <View style={styles.container}>
      <Overlay isVisible={isLoading} fullScreen={true} overlayStyle={{alignItems: "center", justifyContent: 'center'}}>
        <View>
          <Text style={{marginBottom:10}}>{status}</Text>
          <ActivityIndicator size="large" color="lightblue" />
        </View>
      </Overlay>
      
      <Card containerStyle={{width:"100%",marginBottom:10,borderRadius:5}}>
        <Card.Title style={{fontSize:16}}>Image Classification</Card.Title>
        <Card.Divider/>
        <View style={{flexDirection:"row"}}>
          {classList.map((item, key) => {
            return (
              <View style={{flex:1,padding:5}} key={item.id}>
                <Button 
                  title={`${item.name} (${countClassExamples[item.name]})`}
                  onPress={()=>{collectData(item.name)}}
                />
              </View>
            );
          })}
        </View>
      </Card>
      <View style={{width:224,height:224}}>
        {hasPermission && 
          <Camera 
              style={{ flex: 1 }} 
              type={type} 
              ref={ref => {this.camera = ref; }}>
          </Camera>
        }
        {!hasPermission && 
          <View style={{flex:1,alignItems:"center",justifyContent:"center"}}>
            <Text>No camera premission granted</Text>
          </View>
        }
      </View>

      <View style={{flexDirection:"row",padding:5}}>
        <View style={{flex:1,padding:5}}>
          <Button 
            title="Predict"
            onPress={()=>{getPredictions()}}
            disabled={countExamples==0}
          />
        </View>
        <View style={{flex:2,padding:5}}>
          <Text style={{borderRadius:5,borderWidth:1,padding:10,borderColor:"lightgrey"}}>
            {prediction.label}
          </Text>
        </View>
      </View>
    </View>
  );
}
const screenWidth = Math.round(Dimensions.get('window').width);
const screenHeight = Math.round(Dimensions.get('window').height);
const styles = StyleSheet.create({
  container: {
    flexDirection:"column",
    flex:1,
    alignItems: "center",
    justifyContent: 'center',
    padding:10
  },

});
