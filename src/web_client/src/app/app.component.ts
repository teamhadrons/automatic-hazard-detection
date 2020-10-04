import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'hazard-detection-client';

  ngOnInit(): void {
    this.setUpMapContainerHeight();
    const initiatedMap: google.maps.Map = this.initMap();
    this.initiateDrawingManager(initiatedMap);
  }

  initMap(): google.maps.Map {
    const map: google.maps.Map = new google.maps.Map(document.getElementById("map") as HTMLElement, {
      center: { lat: -1.9434858, lng: 30.0594883 },
      zoom: 8
    });
    return map;
  }

  initiateDrawingManager(map: google.maps.Map): void {
    const drawingManager: google.maps.drawing.DrawingManager = new google.maps.drawing.DrawingManager({
      drawingMode: google.maps.drawing.OverlayType.MARKER,
      drawingControl: true,
      drawingControlOptions: {
        position: google.maps.ControlPosition.TOP_CENTER,
        drawingModes: [
          google.maps.drawing.OverlayType.CIRCLE,
          google.maps.drawing.OverlayType.RECTANGLE
        ]
      }
    });
    drawingManager.setMap(map);

    google.maps.event.addListener(drawingManager, 'overlaycomplete', function(e) {
      drawingManager.setDrawingMode(null);
      let bounds = e.overlay.getBounds();
      let northEast = bounds.getNorthEast();
      let southWest = bounds.getSouthWest();
      let center = bounds.getCenter();

      const marker: google.maps.Marker = new google.maps.Marker({
        position: {lat: center.lat(), lng: center.lng()},
        map,
      });

      const infowindowContent: string = 
        `<b> PREDICTION RESULTS </b><br>` +
        `<p><b>North East (Lat):</b> ${northEast.lat()}<br></p>` + 
        `<p><b>North East (Lng):</b> ${northEast.lng()}<br></p>` +
        `<p><b>South West (Lat):</b> ${southWest.lat()}<br></p>` +
        `<p><b>South West (Lng):</b> ${southWest.lng()}<br></p>` +
        ``;

      const infowindow: google.maps.InfoWindow = new google.maps.InfoWindow({
        content: infowindowContent,
      });

      marker.addListener("click", () => {
        infowindow.open(map, marker);
      });
    });
  }

  setUpMapContainerHeight(): void {
    let windowHeight: number = window.innerHeight;
    let mapContainer: HTMLElement = document.getElementById('map-container');
    let toolBar: HTMLElement = document.getElementById('app-component-toolbar');

    // Set map container height.
    mapContainer.style.height = `${windowHeight - toolBar.clientHeight}px`; 
  }
}
