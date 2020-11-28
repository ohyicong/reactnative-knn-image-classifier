import React, { useState, useEffect }  from 'react';
import { StyleSheet, View,Image,TouchableOpacity,Dimensions,ActivityIndicator } from 'react-native';
import { Text,Button, Input,Card,Overlay  } from 'react-native-elements';
import { Camera } from 'expo-camera';
import * as Permissions from 'expo-permissions';
import * as FileSystem from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as jpeg from 'jpeg-js'
export default function App() {
  const datasetLocation= FileSystem.documentDirectory + "dataset.txt";
  const [hasPermission, setHasPermission] = useState(null);
  const [type, setType] = useState(Camera.Constants.Type.back);
  const [isTfReady,setIsTfReady] = useState(false)
  const [mobilenetModel,setMobilenetModel] = useState(null)
  const [knnClassifierModel,setKnnClassifierModel] = useState(null)
  const [dataset,setDataset] = useState(null)
  const [prediction,setPrediction] = useState({
    "label":"No Results",
    "confidence":{}
  })
  const [status,setStatus]=useState("Preparing Model...")
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
        setStatus("Loading Model...")
        setIsLoading(true)
        let { status } = await Permissions.askAsync(Permissions.CAMERA,Permissions.CAMERA_ROLL);
        setHasPermission(status === 'granted');
        console.log("[+] Permission granted")
        await tf.ready()
        setIsTfReady(true)
        setMobilenetModel(await mobilenet.load())
        setKnnClassifierModel(await knnClassifier.create())
        setIsLoading(false)
        console.log("[+] TF Model Loaded")
      }
      setIsLoading(false)
    }
    startup()
  },[isTfReady]);

  //(Part 1) 1. collect and label images from camera
  const collectData = async(className)=>{
    console.log(`[+] Class ${className} selected`)
    setStatus("Training Model...")
    setIsLoading(true)
    try{
      if(this.camera){
        let photo = await this.camera.takePictureAsync({
          skipProcessing: true,
        });
        //(Part 1) 2. resize images into width:224 height:224
        image = await resizeImage(photo.uri, 224 , 224);
        let imageTensor = base64ImageToTensor(image.base64);
        //(Part 1) 3. get embeddings from mobilenet
        let embeddings = await mobilenetModel.infer(imageTensor, true);
        //(Part 1) 4. train knn classifier
        knnClassifierModel.addExample(embeddings,className)
        updateCount(knnClassifierModel)
        console.log("[+] Class Added")
      }
    }catch{
      console.log("[-] No Camera")
      setIsLoading(false)
    }
    
    setIsLoading(false)
  } 
  //(Part 1) 5. predict new images
  const getPredictions = async() =>{
    console.log("[+] Analysing Photo")
    setStatus("Analysing Photo...")
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
        console.log("[+] Prediction: ",JSON.stringify(prediction))
        setPrediction(prediction)
        
      }
    }
    catch(e){
      console.log("[-] No Camera",e)
      setIsLoading(false)
    }
    setIsLoading(false)
    console.log("[+] Photo Analysed")
  }
  //Reset model
  const resetKnnClassifierModel = async() =>{
    console.log("[+] Resetting Model")
    setStatus("Resetting Model...")
    setIsLoading(true)
    try{
      await knnClassifierModel.clearAllClasses()
      setCountExamples(0)
      setCountClassExamples({
        "Class A":0,
        "Class B":0,
        "Class C":0   
      })
    }
    catch{
      console.log("[-] Unable to reset model")
      setIsLoading(false)
    }
    setIsLoading(false)
    console.log("[+] Reset Completed")
  }
  const saveKnnClassifierModel = async() =>{
    console.log("[+] Saving Model")
    setStatus("Saving Model...")
    setIsLoading(true)
    try{
      //Get Dataset
      let dataset = knnClassifierModel.getClassifierDataset()
      //(Part 2) 1. Convert dataset to "string" using JSON.stringify
      let stringDataset=JSON.stringify( Object.entries(dataset).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]))
      setDataset(stringDataset)
      //(Part 2) 2. Save dataset into local file
      await FileSystem.writeAsStringAsync(datasetLocation, stringDataset, { encoding: FileSystem.EncodingType.UTF8 });
      let asset = await MediaLibrary.createAssetAsync(datasetLocation)
      await MediaLibrary.createAlbumAsync("Download", asset, false)
    }
    catch{
      console.log("[-] Unable to save model")
      setIsLoading(false)
    }
    setIsLoading(false)
    console.log("[+] Save Completed")
  }
  const loadKnnClassifierModel = async() =>{
    console.log("[+] Loading Model")
    setStatus("Loading Model...")
    setIsLoading(true)
    try{
      
      //(Part 2) 3. Load dataset from local file
      let stringDataset = await FileSystem.readAsStringAsync(datasetLocation, { encoding: FileSystem.EncodingType.UTF8 })
      let tempModel = knnClassifier.create();
      //(Part 2) 4. Convert dataset format to "JSON" using JSON.parse
      tempModel.setClassifierDataset( Object.fromEntries( JSON.parse(stringDataset).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );
      //(Part 2) 5. Load model
      setKnnClassifierModel(tempModel)
      setDataset(stringDataset)
      updateCount(tempModel)
    }
    catch{
      console.log("[-] Unable to load model")
      setIsLoading(false)
    }
    setIsLoading(false)
    console.log("[+] Load Completed")
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
  function updateCount(tempModel){
    console.log("[+] Update Dataset Count")
    setCountExamples(1)
    var elements = tempModel.getClassExampleCount()
    for (let key in elements){
      if (elements.hasOwnProperty(key)) {
        countClassExamples[key]=elements[key];
      }
    }
    Object.entries(tempModel.getClassExampleCount()).map((key,item)=>{
      let temp = countClassExamples
      temp[key] = item
      setCountClassExamples(temp)
    })
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
      <Card containerStyle={{width:"100%",marginBottom:10,borderRadius:5}}>
        <View style={{flexDirection:"row",padding:5}}>
          <View style={{flex:1,padding:5}}>
            <Button 
              title="Load Model"
              onPress={()=>{loadKnnClassifierModel()}}
              disabled={hasPermission===false}
              type="outline"
            />
          </View>
          <View style={{flex:1,padding:5}}>
            <Button 
              title="Save Model"
              onPress={()=>{saveKnnClassifierModel()}}
              disabled={countExamples==0}
              type="outline"
            />
          </View>
          <View style={{flex:1,padding:5}}>
            <Button 
              title="Reset Model"
              onPress={()=>{resetKnnClassifierModel()}}
              disabled={countExamples==0||hasPermission===false}
              type="outline"
            />
          </View>
        </View>
      </Card>
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
